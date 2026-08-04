"""
Retrieval orchestration.

The retrieval service coordinates query embedding and vector search while
remaining independent from any specific vector database implementation.
"""

from __future__ import annotations

import logging

from anyio import to_thread

from ..core import RetrievalError
from ..embedders import BaseEmbeddingProvider
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
                    chunk_index=result.chunk_index,
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
    top_k: int,
    provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    collection_name: str,
    *,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Run the retrieval pipeline.

    Steps:
        1. Embed the query.
        2. Search the vector store.
        3. Return provider-agnostic domain results — the caller (API
           layer) wraps these into the wire response schema.
    """
    try:
        query_vector = await to_thread.run_sync(
            provider.embed_query,
            query,
        )
    except Exception:
        logger.exception("Query embedding failed")
        raise

    if not query_vector:
        raise RetrievalError("Embedding provider returned an empty query vector.")

    raw_hits = await vector_store.search(
        collection_name=collection_name,
        vector=query_vector,
        top_k=top_k,
        min_score=min_score,
    )
    return _deduplicate_parents(raw_hits)


class RetrievalServiceAdapter:
    """Adapter implementing the RetrievalServicePort interface."""

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

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        query_vector = await to_thread.run_sync(
            self._provider.embed_query,
            query,
        )

        if not query_vector:
            raise RetrievalError("Embedding provider returned an empty query vector.")

        raw_hits = await self._vector_store.search(
            collection_name=self._collection_name,
            vector=query_vector,
            top_k=top_k,
            min_score=self._min_score,
        )
        return _deduplicate_parents(raw_hits)
