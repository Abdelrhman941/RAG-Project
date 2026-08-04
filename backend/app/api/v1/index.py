from uuid import UUID

from fastapi import APIRouter, status

from ...core import ChunkingStrategy
from ...schemas import IndexingResponse
from ...services import index_document
from ..deps import EmbeddingProviderDep, SettingsDep, VectorStoreDep

index_router = APIRouter(prefix="/documents", tags=["Documents"])


@index_router.post(
    "/{document_id}/index",
    status_code=status.HTTP_201_CREATED,
    response_model=IndexingResponse,
)
async def index_uploaded_document(
    document_id: UUID,
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
    vector_store: VectorStoreDep,
) -> IndexingResponse:
    """Parse -> Chunk -> Embed -> Upsert into the configured vector store."""
    return await index_document(
        document_id=document_id,
        upload_dir=settings.UPLOAD_DIR,
        provider=provider,
        vector_store=vector_store,
        collection_name=settings.QDRANT_COLLECTION,
        distance=settings.DISTANCE_METRIC,
        strategy=ChunkingStrategy.TOKEN,
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        overlap=settings.DEFAULT_CHUNK_OVERLAP,
    )
