"""Unit tests for `retrieve` (the RetrievalService)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.core import RetrievalError
from app.core.enums.vector_store import DistanceMetric
from app.embedders import BaseEmbeddingProvider
from app.retrieval import SearchResult
from app.schemas import PointData, RetrievalResponse
from app.services import retrieve
from app.vectorstores import BaseVectorStore


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model_name: str = "fake-model",
        dimension: int = 3,
        vector: list[float] | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._vector = [0.1, 0.2, 0.3] if vector is None else vector
        self.embed_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [list(self._vector) for _ in texts]


class FakeVectorStore(BaseVectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.search_calls: list[dict[str, object]] = []

    async def collection_exists(self, collection_name: str) -> bool:  # pragma: no cover
        return True

    async def create_collection(
        self, collection_name: str, dimension: int, distance: DistanceMetric
    ) -> None:  # pragma: no cover
        return None

    async def upsert(self, collection_name: str, points: Sequence[PointData]) -> int:
        return 0

    async def delete_by_document(
        self, collection_name: str, document_id: str
    ) -> None:  # pragma: no cover
        return None

    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "vector": list(vector),
                "top_k": top_k,
                "min_score": min_score,
            }
        )
        return list(self._results)

    async def health_check(self) -> bool:  # pragma: no cover
        return True


def _hit(score: float, content: str = "hit") -> SearchResult:
    return SearchResult(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        score=score,
        content=content,
    )


@pytest.mark.anyio
class TestRetrieveChunks:
    async def test_returns_ranked_results_from_vector_store(self) -> None:
        hits = [_hit(0.95, "top"), _hit(0.80, "second"), _hit(0.70, "third")]
        provider = FakeEmbeddingProvider(model_name="model-x")
        store = FakeVectorStore(results=hits)

        response = await retrieve(
            query="What is DBSCAN?",
            top_k=3,
            provider=provider,
            vector_store=store,
            collection_name="documents",
            min_score=0.3,
        )

        assert isinstance(response, RetrievalResponse)
        assert response.query == "What is DBSCAN?"
        assert response.embedding_model == "model-x"
        assert response.total_results == 3
        # Ranking is preserved (never re-sorted by the service).
        assert [r.content for r in response.results] == ["top", "second", "third"]

    async def test_query_is_forwarded_verbatim_for_embedding(self) -> None:
        provider = FakeEmbeddingProvider()
        store = FakeVectorStore(results=[])

        await retrieve(
            query="hello world",
            top_k=5,
            provider=provider,
            vector_store=store,
            collection_name="documents",
        )

        assert provider.embed_calls == [["hello world"]]

    async def test_top_k_and_min_score_are_forwarded_to_store(self) -> None:
        provider = FakeEmbeddingProvider()
        store = FakeVectorStore(results=[])

        await retrieve(
            query="q",
            top_k=7,
            provider=provider,
            vector_store=store,
            collection_name="my-collection",
            min_score=0.5,
        )

        assert store.search_calls == [
            {
                "collection_name": "my-collection",
                "vector": [0.1, 0.2, 0.3],
                "top_k": 7,
                "min_score": 0.5,
            }
        ]

    async def test_empty_results_produce_empty_response(self) -> None:
        provider = FakeEmbeddingProvider()
        store = FakeVectorStore(results=[])

        response = await retrieve(
            query="nothing",
            top_k=5,
            provider=provider,
            vector_store=store,
            collection_name="documents",
        )

        assert response.total_results == 0
        assert response.results == []

    async def test_empty_vector_from_provider_raises_retrieval_error(self) -> None:
        provider = FakeEmbeddingProvider(vector=[])
        store = FakeVectorStore(results=[])

        with pytest.raises(RetrievalError):
            await retrieve(
                query="q",
                top_k=5,
                provider=provider,
                vector_store=store,
                collection_name="documents",
            )

    async def test_result_ids_are_uuids(self) -> None:
        hits = [_hit(0.9)]
        provider = FakeEmbeddingProvider()
        store = FakeVectorStore(results=hits)

        response = await retrieve(
            query="q",
            top_k=1,
            provider=provider,
            vector_store=store,
            collection_name="documents",
        )

        assert isinstance(response.results[0].document_id, UUID)
        assert isinstance(response.results[0].chunk_id, UUID)
