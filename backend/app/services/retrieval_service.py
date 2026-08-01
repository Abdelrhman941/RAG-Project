import logging
from collections.abc import Sequence

from anyio import to_thread

from ..core import RetrievalError
from ..embedders import BaseEmbeddingProvider
from ..retrieval import SearchResult
from ..schemas import RetrievalResponse, RetrievedChunk
from ..vectorstores import BaseVectorStore

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    top_k: int,
    provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    collection_name: str,
    *,
    min_score: float | None = None,
) -> RetrievalResponse:
    """Run the retrieval pipeline: embed the query, then semantic-search.

    Pure orchestration:
        1. Receive the query.
        2. Generate its embedding (sync CPU work, pushed off the loop).
        3. Delegate the similarity search to `vector_store.search()`.
        4. Wrap the domain-level `SearchResult`s in the public
            `RetrievalResponse` schema.

    The service NEVER imports Qdrant — vendor concerns stay inside the
    adapter, HTTP concerns stay inside the API layer. Ranking is done
    by the vector store; we never reorder here.
    """
    # 1) Embed the query. The embedding call is sync CPU-bound; run it
    # in the default worker thread the same way `index_document` does.
    try:
        vectors = await to_thread.run_sync(provider.embed, [query])
    except Exception:
        logger.exception("Query embedding failed")
        raise

    if not vectors or not vectors[0]:
        # Defensive: an embedding provider that returns nothing for a
        # non-empty query is a bug we should surface, not swallow.
        raise RetrievalError("Embedding provider returned an empty vector.")

    query_vector: Sequence[float] = vectors[0]

    if not query_vector:
        raise RetrievalError("Query embedding is empty.")

    # 2) Similarity search. The store is responsible for ranking and
    # for applying `min_score` server-side.
    hits: list[SearchResult] = await vector_store.search(
        collection_name=collection_name,
        vector=query_vector,
        top_k=top_k,
        min_score=min_score,
    )

    # 3) Map domain models -> public response schema. The two shapes
    # are deliberately kept separate: `SearchResult` is internal, while
    # `RetrievedChunk` is the wire contract clients depend on.
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
        query=query,
        embedding_model=provider.model_name,
        total_results=len(results),
        results=results,
    )
