from fastapi import APIRouter, status

from ...schemas import RetrievalRequest, RetrievalResponse, RetrievedChunk
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
    # `TopKQueryRequest._resolve_top_k` guarantees this is never None once
    # the request object exists — the field stays `int | None` in the
    # schema only because pydantic can't narrow a validator's effect
    # statically.
    assert request.top_k is not None
    top_k: int = request.top_k

    hits = await retrieve(
        query=request.query,
        top_k=top_k,
        provider=provider,
        vector_store=vector_store,
        collection_name=settings.QDRANT_COLLECTION,
        min_score=settings.MIN_SCORE,
    )

    results = [
        RetrievedChunk(
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            chunk_index=hit.chunk_index,
            page_number=hit.page_number,
            score=hit.score,
            content=hit.content,
        )
        for hit in hits
    ]

    return RetrievalResponse(
        query=request.query,
        embedding_model=provider.model_name,
        total_results=len(results),
        results=results,
    )
