import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, File, UploadFile, status

from ...core import (
    ChunkingStrategy,
    DocumentExtension,
    DocumentStatus,
    UnsupportedDocumentTypeError,
)
from ...schemas import (
    ChunkRequest,
    ChunkResponse,
    DocumentUploadResponse,
    IndexingResponse,
    ParsedDocument,
)
from ..deps import (
    DocumentChunkerServiceDep,
    DocumentIndexerServiceDep,
    DocumentParserServiceDep,
    FileStorageServiceDep,
    SettingsDep,
)

documents_router = APIRouter(prefix="/documents", tags=["Documents"])


@documents_router.post(
    "", status_code=status.HTTP_201_CREATED, response_model=DocumentUploadResponse
)
async def upload_document(
    settings: SettingsDep,
    storage: FileStorageServiceDep,
    file: Annotated[
        UploadFile,
        File(description="Upload a .pdf, .txt or .md document"),
    ],
) -> DocumentUploadResponse:
    # --- Validation 1: Check if a file was actually selected and sent ---
    if not file.filename:
        raise UnsupportedDocumentTypeError(
            "", allowed=[item.value for item in DocumentExtension]
        )

    # --- Validation 2: Extract the extension and verify it against allowed types ---
    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()
    try:
        extension_enum = DocumentExtension(extension)
    except ValueError:
        raise UnsupportedDocumentTypeError(
            extension, allowed=[item.value for item in DocumentExtension]
        ) from None

    file_id = uuid.uuid4()
    filename = f"{file_id}{extension_enum.value}"
    destination = settings.UPLOAD_DIR / filename

    # --- Validation 3: Save the file locally while validating the max file size ---
    size_bytes = await storage.save_uploaded_file(
        file=file,
        destination=destination,
    )

    return DocumentUploadResponse(
        id=file_id,
        filename=filename,
        original_filename=original_filename,
        extension=extension_enum,
        size_bytes=size_bytes,
        status=DocumentStatus.UPLOADED,
        uploaded_at=datetime.now(timezone.utc),
    )


@documents_router.post(
    "/{document_id}/parse",
    status_code=status.HTTP_200_OK,
    response_model=ParsedDocument,
)
async def parse_uploaded_document(
    document_id: uuid.UUID,
    parser: DocumentParserServiceDep,
) -> ParsedDocument:
    async with parser.parse_document(document_id=document_id) as (
        pages_gen,
        source_type,
    ):
        pages = [p async for p in pages_gen]

    return ParsedDocument(
        document_id=document_id,
        status=DocumentStatus.PARSING,
        pages=pages,
    )


@documents_router.post(
    "/{document_id}/chunks",
    status_code=status.HTTP_201_CREATED,
    response_model=ChunkResponse,
)
async def chunk_uploaded_document(
    document_id: uuid.UUID,
    settings: SettingsDep,
    chunker: DocumentChunkerServiceDep,
    chunk_request: Annotated[ChunkRequest, Body(default_factory=ChunkRequest)],
) -> ChunkResponse:
    embedding_chunk_size = (
        chunk_request.embedding_chunk_size
        if chunk_request.embedding_chunk_size is not None
        else settings.DEFAULT_EMBEDDING_CHUNK_SIZE
    )
    embedding_overlap = (
        chunk_request.embedding_overlap
        if chunk_request.embedding_overlap is not None
        else settings.DEFAULT_EMBEDDING_OVERLAP
    )
    prompt_chunk_size = (
        chunk_request.prompt_chunk_size
        if chunk_request.prompt_chunk_size is not None
        else settings.DEFAULT_PROMPT_CHUNK_SIZE
    )
    prompt_overlap = (
        chunk_request.prompt_overlap
        if chunk_request.prompt_overlap is not None
        else settings.DEFAULT_PROMPT_OVERLAP
    )

    chunks = [
        c
        async for c in chunker.chunk_document(
            document_id=document_id,
            strategy=chunk_request.strategy,
            embedding_chunk_size=embedding_chunk_size,
            embedding_overlap=embedding_overlap,
            prompt_chunk_size=prompt_chunk_size,
            prompt_overlap=prompt_overlap,
        )
    ]

    return ChunkResponse(
        document_id=document_id,
        status=DocumentStatus.CHUNKING,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@documents_router.post(
    "/{document_id}/index",
    status_code=status.HTTP_201_CREATED,
    response_model=IndexingResponse,
)
async def index_uploaded_document(
    document_id: uuid.UUID,
    settings: SettingsDep,
    indexer: DocumentIndexerServiceDep,
) -> IndexingResponse:
    """Parse -> Chunk -> Embed -> Upsert into the configured vector store."""
    return await indexer.index_document(
        document_id=document_id,
        collection_name=settings.QDRANT_COLLECTION,
        distance=settings.DISTANCE_METRIC,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=settings.DEFAULT_EMBEDDING_CHUNK_SIZE,
        embedding_overlap=settings.DEFAULT_EMBEDDING_OVERLAP,
        prompt_chunk_size=settings.DEFAULT_PROMPT_CHUNK_SIZE,
        prompt_overlap=settings.DEFAULT_PROMPT_OVERLAP,
    )
