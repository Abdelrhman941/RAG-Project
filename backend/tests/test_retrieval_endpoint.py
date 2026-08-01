"""Integration tests for `POST /api/v1/retrieval/search`.

The Qdrant client is replaced end-to-end with a fake `BaseVectorStore`
via FastAPI's dependency-override mechanism, so these tests exercise
the full HTTP -> Service -> Store pipeline without any network I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_embedding_provider, get_current_vector_store
from app.core import CollectionNotFoundError
from app.core.enums.vector_store import DistanceMetric
from app.embedders import BaseEmbeddingProvider
from app.main import app
from app.retrieval import SearchResult
from app.schemas.point import PointData
from app.vectorstores import BaseVectorStore


# ---------- Fakes ----------
class StubEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "stub-embedding-model"

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubVectorStore(BaseVectorStore):
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        raise_not_found: bool = False,
    ) -> None:
        self._results = results or []
        self._raise_not_found = raise_not_found
        self.last_top_k: int | None = None

    async def collection_exists(self, collection_name: str) -> bool:
        return True

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: DistanceMetric,
    ) -> None:
        return None

    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointData],
    ) -> int:
        return 0

    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        return None

    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        self.last_top_k = top_k
        if self._raise_not_found:
            raise CollectionNotFoundError(collection_name)
        return list(self._results[:top_k])

    async def health_check(self) -> bool:  # pragma: no cover
        return True


# ---------- Fixtures ----------
@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _override(provider: BaseEmbeddingProvider, store: BaseVectorStore) -> None:
    app.dependency_overrides[get_current_embedding_provider] = lambda: provider
    app.dependency_overrides[get_current_vector_store] = lambda: store


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _hit(score: float, content: str = "hit", chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=chunk_index,
        page_number=1,
        score=score,
        content=content,
    )


# ---------- Tests ----------
class TestSearchEndpoint:
    def test_returns_ranked_top_k_chunks(self, client: TestClient) -> None:
        hits = [
            _hit(0.95, "top", 0),
            _hit(0.80, "second", 1),
            _hit(0.70, "third", 2),
        ]
        _override(StubEmbeddingProvider(), StubVectorStore(results=hits))

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "What is DBSCAN?", "top_k": 3},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "What is DBSCAN?"
        assert body["embedding_model"] == "stub-embedding-model"
        assert body["total_results"] == 3
        assert [r["content"] for r in body["results"]] == ["top", "second", "third"]
        # Ranking is preserved by the endpoint / service.
        assert body["results"][0]["score"] >= body["results"][1]["score"]
        assert body["results"][1]["score"] >= body["results"][2]["score"]

    def test_top_k_is_respected(self, client: TestClient) -> None:
        hits = [_hit(0.9 - i * 0.05, f"hit-{i}", i) for i in range(10)]
        store = StubVectorStore(results=hits)
        _override(StubEmbeddingProvider(), store)

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "clustering", "top_k": 4},
        )

        assert response.status_code == 200
        assert store.last_top_k == 4
        assert response.json()["total_results"] == 4

    def test_empty_results_are_returned_as_empty_list(self, client: TestClient) -> None:
        _override(StubEmbeddingProvider(), StubVectorStore(results=[]))

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "nothing matches", "top_k": 5},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_results"] == 0
        assert body["results"] == []

    def test_top_k_defaults_when_omitted(self, client: TestClient) -> None:
        store = StubVectorStore(results=[])
        _override(StubEmbeddingProvider(), store)

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "hello"},
        )

        assert response.status_code == 200
        assert store.last_top_k == 5  # DEFAULT_TOP_K

    def test_missing_query_returns_422(self, client: TestClient) -> None:
        _override(StubEmbeddingProvider(), StubVectorStore())
        response = client.post("/api/v1/retrieval/search", json={"top_k": 3})
        assert response.status_code == 422

    def test_blank_query_returns_422(self, client: TestClient) -> None:
        _override(StubEmbeddingProvider(), StubVectorStore())
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "   ", "top_k": 3},
        )
        assert response.status_code == 422

    def test_top_k_above_maximum_returns_422(self, client: TestClient) -> None:
        _override(StubEmbeddingProvider(), StubVectorStore())
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "hello", "top_k": 999},
        )
        assert response.status_code == 422

    def test_top_k_below_minimum_returns_422(self, client: TestClient) -> None:
        _override(StubEmbeddingProvider(), StubVectorStore())
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "hello", "top_k": 0},
        )
        assert response.status_code == 422

    def test_missing_collection_returns_404(self, client: TestClient) -> None:
        _override(
            StubEmbeddingProvider(),
            StubVectorStore(raise_not_found=True),
        )
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "hello", "top_k": 3},
        )
        assert response.status_code == 404

    def test_response_shape_matches_schema(self, client: TestClient) -> None:
        hits = [_hit(0.91, "DBSCAN is a density-based clustering algorithm...", 18)]
        _override(StubEmbeddingProvider(), StubVectorStore(results=hits))

        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "What is DBSCAN?", "top_k": 5},
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "query",
            "embedding_model",
            "total_results",
            "results",
        }
        result = body["results"][0]
        assert set(result.keys()) == {
            "document_id",
            "chunk_id",
            "chunk_index",
            "page_number",
            "score",
            "content",
        }
