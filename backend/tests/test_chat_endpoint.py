"""
Integration tests for `POST /api/v1/chat`.

Follows the same pattern as `test_retrieval_endpoint.py`: embedding
provider, vector store, and (new here) the LLM provider are all
replaced via FastAPI's dependency-override mechanism, so these tests
exercise the full HTTP -> GenerationService -> RetrievalServiceAdapter
-> PromptBuilder -> LLM pipeline without any real network I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_embedding_provider, get_current_vector_store
from app.core import CollectionNotFoundError
from app.core.enums.vector_store import DistanceMetric
from app.core.exceptions import LLMProviderError
from app.embedders import BaseEmbeddingProvider
from app.llms import BaseLLMProvider, get_llm_provider
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


class StubLLMProvider(BaseLLMProvider):
    """Stands in for GroqProvider — no network, deterministic output."""

    def __init__(
        self,
        answer: str = "Paris is the capital of France.",
        *,
        raise_error: bool = False,
    ) -> None:
        self._answer = answer
        self._raise_error = raise_error
        self.last_messages: list[Any] | None = None

    @property
    def model_name(self) -> str:
        return "stub-llm-model"

    async def generate(
        self,
        messages: Sequence[Any],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_messages = list(messages)
        if self._raise_error:
            raise LLMProviderError("Groq request failed")
        return self._answer


# ---------- Fixtures ----------
@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _override(
    embedding_provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    llm_provider: BaseLLMProvider,
) -> None:
    app.dependency_overrides[get_current_embedding_provider] = lambda: (
        embedding_provider
    )
    app.dependency_overrides[get_current_vector_store] = lambda: vector_store
    app.dependency_overrides[get_llm_provider] = lambda: llm_provider


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _hit(
    score: float, content: str = "Paris is the capital of France."
) -> SearchResult:
    return SearchResult(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        score=score,
        content=content,
    )


# ---------- Tests ----------
def test_valid_query_returns_answer(client: TestClient) -> None:
    _override(StubEmbeddingProvider(), StubVectorStore([_hit(0.9)]), StubLLMProvider())

    response = client.post(
        "/api/v1/chat", json={"query": "What is the capital of France?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is the capital of France."
    assert body["query"] == "What is the capital of France?"
    assert body["model"] == "stub-llm-model"


def test_response_includes_citations(client: TestClient) -> None:
    hits = [_hit(0.91), _hit(0.77)]
    _override(StubEmbeddingProvider(), StubVectorStore(hits), StubLLMProvider())

    response = client.post(
        "/api/v1/chat", json={"query": "What is the capital of France?"}
    )

    body = response.json()
    assert body["total_citations"] == 2
    assert len(body["citations"]) == 2
    assert body["citations"][0]["document_id"] == str(hits[0].document_id)
    assert body["citations"][0]["score"] == pytest.approx(0.91)


def test_response_schema_is_correct(client: TestClient) -> None:
    _override(StubEmbeddingProvider(), StubVectorStore([_hit(0.9)]), StubLLMProvider())

    response = client.post(
        "/api/v1/chat", json={"query": "What is the capital of France?"}
    )

    body = response.json()
    assert set(body.keys()) == {
        "query",
        "answer",
        "model",
        "total_citations",
        "citations",
    }
    for citation in body["citations"]:
        assert set(citation.keys()) == {
            "document_id",
            "chunk_id",
            "page_number",
            "score",
        }


def test_invalid_query_returns_422(client: TestClient) -> None:
    _override(StubEmbeddingProvider(), StubVectorStore([]), StubLLMProvider())

    response = client.post("/api/v1/chat", json={"query": "   "})

    assert response.status_code == 422


def test_missing_query_returns_422(client: TestClient) -> None:
    _override(StubEmbeddingProvider(), StubVectorStore([]), StubLLMProvider())

    response = client.post("/api/v1/chat", json={})

    assert response.status_code == 422


def test_llm_failure_returns_500(client: TestClient) -> None:
    _override(
        StubEmbeddingProvider(),
        StubVectorStore([_hit(0.9)]),
        StubLLMProvider(raise_error=True),
    )

    response = client.post(
        "/api/v1/chat", json={"query": "What is the capital of France?"}
    )

    assert response.status_code == 500


def test_retrieval_failure_bubbles_correctly(client: TestClient) -> None:
    _override(
        StubEmbeddingProvider(),
        StubVectorStore(raise_not_found=True),
        StubLLMProvider(),
    )

    response = client.post(
        "/api/v1/chat", json={"query": "What is the capital of France?"}
    )

    # CollectionNotFoundError propagates through RetrievalServiceAdapter ->
    # GenerationService untouched, and is handled by the same global
    # handler the retrieval endpoint uses.
    assert response.status_code == 404
