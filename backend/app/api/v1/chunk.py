from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, status

from ...core import DocumentStatus
from ...schemas import ChunkRequest, ChunkResponse
from ...services import chunk_document
from ..deps import SettingsDep

chunk_router = APIRouter(prefix="/documents", tags=["Documents"])


@chunk_router.post(
    "/{document_id}/chunks",
    status_code=status.HTTP_201_CREATED,
    response_model=ChunkResponse,
)
async def chunk_uploaded_document(
    document_id: UUID,
    settings: SettingsDep,
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

    chunks_gen = chunk_document(
        document_id=document_id,
        upload_dir=settings.UPLOAD_DIR,
        strategy=chunk_request.strategy,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        prompt_chunk_size=prompt_chunk_size,
        prompt_overlap=prompt_overlap,
    )
    chunks = [c async for c in chunks_gen]

    return ChunkResponse(
        document_id=document_id,
        status=DocumentStatus.CHUNKING,
        total_chunks=len(chunks),
        chunks=chunks,
    )
