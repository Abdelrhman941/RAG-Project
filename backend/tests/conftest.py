import importlib
import io
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.core import get_settings
from app.core.enums.vector_store import DistanceMetric
from app.embedders import BaseEmbeddingProvider
from app.retrieval import SearchResult
from app.schemas import PointData
from app.vectorstores import BaseVectorStore


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, near-instant stand-in for SentenceTransformerProvider."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store used to test the indexing pipeline without Qdrant."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.points: dict[str, list[PointData]] = defaultdict(list)
        self.deleted_documents: list[tuple[str, str]] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: DistanceMetric,
    ) -> None:
        if collection_name in self.collections:
            return
        self.collections[collection_name] = {
            "dimension": dimension,
            "distance": distance,
        }

    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        return []

    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointData],
    ) -> int:
        self.points[collection_name].extend(points)
        return len(points)

    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        self.deleted_documents.append((collection_name, document_id))
        self.points[collection_name] = [
            point
            for point in self.points[collection_name]
            if str(point.payload.document_id) != document_id
        ]

    async def health_check(self) -> bool:
        return True


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_vector_store: FakeVectorStore,
) -> Iterator[TestClient]:
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")

    get_settings.cache_clear()

    from app import main as main_module
    from app.api.deps import (
        get_current_embedding_provider,
        get_current_vector_store,
    )

    importlib.reload(main_module)

    main_module.app.dependency_overrides[get_current_embedding_provider] = lambda: (
        FakeEmbeddingProvider()
    )
    main_module.app.dependency_overrides[get_current_vector_store] = lambda: (
        fake_vector_store
    )

    with TestClient(main_module.app) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def make_pdf_bytes() -> Callable[[list[str]], bytes]:
    """Factory fixture: build a real PDF with one page per given string of text."""

    def _make(page_texts: list[str]) -> bytes:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        for text in page_texts:
            if text:
                pdf.drawString(72, 720, text)
            pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    return _make
