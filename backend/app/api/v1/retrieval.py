from fastapi import APIRouter, HTTPException, status

from ...schemas import RetrievalRequest, RetrievalResponse, RetrievedChunk
from ..deps import RetrievalServiceDep, SettingsDep

retrieval_router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@retrieval_router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    response_model=RetrievalResponse,
)
async def search(
    request: RetrievalRequest,
    settings: SettingsDep,
    retrieval_service: RetrievalServiceDep,
) -> RetrievalResponse:
    """Semantic search over the configured Qdrant collection.

    Pipeline: query -> embedding -> Qdrant search -> top-k ranked chunks.
    Generation is NOT performed here (that belongs to Sprint 9).
    """
    top_k: int = request.top_k if request.top_k is not None else settings.DEFAULT_TOP_K
    if top_k > settings.MAX_TOP_K:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"top_k must be ≤ {settings.MAX_TOP_K}.",
        )

    hits = await retrieval_service.retrieve(
        query=request.query,
        top_k=top_k,
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
        embedding_model=retrieval_service.embedding_model_name,
        total_results=len(results),
        results=results,
    )
