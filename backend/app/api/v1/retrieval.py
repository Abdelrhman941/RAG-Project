from fastapi import APIRouter, status

from ...schemas import RetrievalRequest, RetrievalResponse
from ...services import retrieve
from ..deps import EmbeddingProviderDep, SettingsDep, VectorStoreDep

retrieval_router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@retrieval_router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    response_model=RetrievalResponse,
)
async def search(
    request: RetrievalRequest,
    settings: SettingsDep,
    provider: EmbeddingProviderDep,
    vector_store: VectorStoreDep,
) -> RetrievalResponse:
    """Semantic search over the configured Qdrant collection.

    Pipeline: query -> embedding -> Qdrant search -> top-k ranked chunks.
    Generation is NOT performed here (that belongs to Sprint 9).
    """
    return await retrieve(
        query=request.query,
        top_k=request.top_k,
        provider=provider,
        vector_store=vector_store,
        collection_name=settings.QDRANT_COLLECTION,
        min_score=settings.MIN_SCORE,
    )
