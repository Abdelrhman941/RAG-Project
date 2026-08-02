"""
Retrieval orchestration.

Two things live here on purpose:

1. `retrieve()` — the original free function (Sprint 7/8). It needs
    FOUR collaborators (embedding provider, vector store, collection
    name, min_score) because it's called directly from the retrieval
    HTTP endpoint, which has all of those via FastAPI dependencies.

2. `RetrievalServiceAdapter` (Sprint 9, Phase 4) — wraps #1 to satisfy
    `generation.ports.RetrievalServicePort`, whose contract is just
    `retrieve(query, top_k) -> list[RetrievedChunk]`.

    Why the adapter exists: `GenerationService` must not know about
    embedding providers, vector stores, or collection names — that's
    retrieval's job, not generation's (Phase 3 rule: "the service must
    reuse the existing RetrievalService"). But there was no class with a
    bound `.retrieve(query, top_k)` method — only the free function
    above, which needs those extra collaborators on every call. The
    adapter binds them once (at DI time, in `api/deps.py`) and exposes
    the narrow interface GenerationService actually depends on.
"""

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
    try:
        vectors = await to_thread.run_sync(provider.embed, [query])
    except Exception:
        logger.exception("Query embedding failed")
        raise

    if not vectors or not vectors[0]:
        raise RetrievalError("Embedding provider returned an empty vector.")

    query_vector: Sequence[float] = vectors[0]

    if not query_vector:
        raise RetrievalError("Query embedding is empty.")

    hits: list[SearchResult] = await vector_store.search(
        collection_name=collection_name,
        vector=query_vector,
        top_k=top_k,
        min_score=min_score,
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
        query=query,
        embedding_model=provider.model_name,
        total_results=len(results),
        results=results,
    )


class RetrievalServiceAdapter:
    """Adapts the free `retrieve()` function to `RetrievalServicePort`.

    Binds the embedding provider, vector store, collection name, and
    min_score once — so `GenerationService` only ever deals with
    `(query, top_k)`, exactly as `generation.ports.RetrievalServicePort`
    declares.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        collection_name: str,
        min_score: float | None = None,
    ) -> None:
        self._provider = provider
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._min_score = min_score

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        response = await retrieve(
            query=query,
            top_k=top_k,
            provider=self._provider,
            vector_store=self._vector_store,
            collection_name=self._collection_name,
            min_score=self._min_score,
        )
        return response.results
