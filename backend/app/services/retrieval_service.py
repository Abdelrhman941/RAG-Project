"""
Retrieval orchestration.

The retrieval service coordinates query embedding and vector search while
remaining independent from any specific vector database implementation.
"""

from __future__ import annotations

import logging

from anyio import to_thread

from ..core import RetrievalError
from ..embedders import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    BaseSparseEmbeddingProvider,
)
from ..generation import RetrievalServicePort
from ..retrieval import SearchResult
from ..vectorstores import BaseVectorStore

logger = logging.getLogger(__name__)


def _deduplicate_parents(results: list[SearchResult]) -> list[SearchResult]:
    """Group results by parent_chunk_id and return deduplicated parents."""
    seen_parents = set()
    deduped = []

    for result in results:
        if result.parent_chunk_id:
            if result.parent_chunk_id in seen_parents:
                continue
            seen_parents.add(result.parent_chunk_id)

            # Create a new SearchResult representing the parent
            deduped.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.parent_chunk_id,
                    chunk_index=0,  # Placeholder for reconstructed parent
                    page_number=result.page_number,
                    score=result.score,
                    content=result.parent_content,  # type: ignore[arg-type]
                )
            )
        else:
            deduped.append(result)

    return deduped


async def retrieve(
    query: str,
    provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    collection_name: str,
    top_k: int = 5,
    fetch_k: int = 15,
    min_score: float | None = None,
    rerank_min_score: float = 0.3,
    sparse_provider: BaseSparseEmbeddingProvider | None = None,
    reranker: BaseRerankerProvider | None = None,
) -> list[SearchResult]:
    """Retrieve the most relevant document chunks for a given query."""
    query_vector = await to_thread.run_sync(provider.embed_query, query)
    if not query_vector:
        raise RetrievalError("Embedding provider returned an empty query vector.")

    sparse_vector = None
    if sparse_provider:
        sparse_vector = await to_thread.run_sync(
            sparse_provider.embed_sparse_query, query
        )

    # 1. Fetch broad candidate set (fetch_k)
    raw_hits = await vector_store.search(
        collection_name=collection_name,
        vector=query_vector,
        top_k=fetch_k if reranker else top_k,
        min_score=min_score if not reranker else None,
        sparse_vector=sparse_vector,
    )

    # 2. Parent Dedup
    deduped = _deduplicate_parents(raw_hits)

    # 3. Rerank if configured
    if not reranker or not deduped:
        return deduped[:top_k]

    texts_to_rerank = [hit.content for hit in deduped]
    rerank_scores = await to_thread.run_sync(reranker.rerank, query, texts_to_rerank)

    # 4. Threshold & Update Scores
    reranked_results = []
    for hit, r_score in zip(deduped, rerank_scores, strict=True):
        if r_score >= rerank_min_score:
            # We preserve the original Qdrant score and add rerank_score
            updated_hit = hit.model_copy(update={"rerank_score": r_score})
            reranked_results.append(updated_hit)

    # 5. Top-K
    reranked_results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
    return reranked_results[:top_k]


class RetrievalServiceAdapter(RetrievalServicePort):
    """Adapter implementing the RetrievalServicePort interface."""

    def __init__(
        self,
        collection_name: str,
        provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        min_score: float = 0.7,
        fetch_k: int = 15,
        rerank_min_score: float = 0.3,
        sparse_provider: BaseSparseEmbeddingProvider | None = None,
        reranker: BaseRerankerProvider | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._provider = provider
        self._vector_store = vector_store
        self._min_score = min_score
        self._fetch_k = fetch_k
        self._rerank_min_score = rerank_min_score
        self._sparse_provider = sparse_provider
        self._reranker = reranker

    @property
    def embedding_model_name(self) -> str:
        """Public accessor for the embedding model name used during retrieval."""
        return self._provider.model_name

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return await retrieve(
            query=query,
            collection_name=self._collection_name,
            provider=self._provider,
            vector_store=self._vector_store,
            top_k=top_k,
            fetch_k=self._fetch_k,
            min_score=self._min_score,
            rerank_min_score=self._rerank_min_score,
            sparse_provider=self._sparse_provider,
            reranker=self._reranker,
        )
