from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core import DistanceMetric
from app.embedders.base import BaseEmbeddingProvider
from app.embedders.factory import get_embedding_provider
from app.main import app
from app.retrieval import SearchResult
from app.vectorstores.base import BaseVectorStore
from app.vectorstores.factory import get_vector_store
from app.vectorstores.models import PointData

client = TestClient(app)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def embedding_dimension(self) -> int:
        return 768

    @property
    def max_sequence_length(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-model"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self.embedding_dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.embedding_dimension


class MockVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.points: dict[str, PointData] = {}

    async def get_existing_hashes(
        self, collection_name: str, candidate_hashes: frozenset[str]
    ) -> frozenset[str]:
        return frozenset()

    async def create_collection(
        self, collection_name: str, dimension: int, distance: DistanceMetric
    ) -> None:
        pass

    async def delete_by_document(self, collection_name: str, document_id: str) -> None:
        pass

    async def upsert(self, collection_name: str, points: Sequence[PointData]) -> int:
        for p in points:
            self.points[str(p.id)] = p
        return len(points)

    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int = 5,
        *,
        min_score: float | None = None,
        sparse_vector: Any = None,
    ) -> list[SearchResult]:

        return [
            SearchResult(
                chunk_id=p.payload.chunk_id,
                document_id=p.payload.document_id,
                chunk_index=p.payload.chunk_index,
                page_number=p.payload.page_number,
                content=p.payload.content,
                score=0.9,
            )
            for p in self.points.values()
        ][:top_k]

    async def mget(
        self, collection_name: str, ids: Sequence[str]
    ) -> list[SearchResult]:
        return []

    async def delete_collection(self, collection_name: str) -> None:
        pass

    async def collection_exists(self, collection_name: str) -> bool:
        return True

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def override_dependencies() -> Any:
    mock_store = MockVectorStore()
    mock_provider = MockEmbeddingProvider()
    app.dependency_overrides[get_vector_store] = lambda: mock_store
    app.dependency_overrides[get_embedding_provider] = lambda: mock_provider
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="This is a test document for the RAG pipeline.", align="C")
    pdf.cell(200, 10, text="It has multiple sentences to test chunking and indexing.")
    pdf.cell(200, 10, text="What is the capital of France? Paris.")

    pdf_path = tmp_path / "test_doc.pdf"
    pdf.output(str(pdf_path))
    return pdf_path


def test_full_rag_pipeline(override_dependencies: Any, sample_pdf: Path) -> None:
    # 1. Upload
    with sample_pdf.open("rb") as f:
        upload_resp = client.post(
            "/api/v1/documents", files={"file": (sample_pdf.name, f, "application/pdf")}
        )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    # 2. Parse (optional if Index does it, but we can verify it works)
    parse_resp = client.post(f"/api/v1/documents/{doc_id}/parse")
    assert parse_resp.status_code == 200
    assert "status" in parse_resp.json()

    # 3. Index (which internally does parse -> chunk -> embed -> upsert)
    index_resp = client.post(f"/api/v1/documents/{doc_id}/index")
    assert index_resp.status_code == 201
    index_data = index_resp.json()
    assert index_data["total_chunks"] > 0
    assert index_data["indexed_points"] > 0

    # 4. Search (Retrieval)
    search_resp = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "capital of France",
            "top_k": 3,
            "min_score": 0.0,
            "collection_name": index_data["collection_name"],
            "distance": DistanceMetric.COSINE.value,
        },
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data["results"]) > 0

    # Verify content
    top_result = search_data["results"][0]
    assert "Paris" in top_result["content"] or "capital" in top_result["content"]
