from uuid import UUID

from fastapi import APIRouter, status

from ...core import ChunkingStrategy
from ...schemas import EmbeddingResponse
from ...services import embed_document
from ..deps import EmbeddingProviderDep, SettingsDep

embed_router = APIRouter(prefix="/documents", tags=["Documents"])


@embed_router.post(
    "/{document_id}/embed",
    status_code=status.HTTP_200_OK,
    response_model=EmbeddingResponse,
)
async def embed_uploaded_document(
    document_id: UUID,
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
) -> EmbeddingResponse:
    return await embed_document(
        document_id=document_id,
        upload_dir=settings.UPLOAD_DIR,
        provider=provider,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=settings.DEFAULT_EMBEDDING_CHUNK_SIZE,
        embedding_overlap=settings.DEFAULT_EMBEDDING_OVERLAP,
    )
