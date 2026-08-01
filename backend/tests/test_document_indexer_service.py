import uuid
from pathlib import Path

import pytest

from app.core import ChunkingStrategy, DistanceMetric, IndexingError
from app.services import index_document
from tests.conftest import FakeEmbeddingProvider, FakeVectorStore


@pytest.mark.anyio
async def test_index_document_orchestrates_full_pipeline(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    document_id = uuid.uuid4()
    (upload_dir / f"{document_id}.txt").write_text(
        "one two three four five " * 100, encoding="utf-8"
    )

    store = FakeVectorStore()
    provider = FakeEmbeddingProvider()

    response = await index_document(
        document_id=document_id,
        upload_dir=upload_dir,
        provider=provider,
        vector_store=store,
        collection_name="documents",
        distance=DistanceMetric.COSINE,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=100,
        overlap=20,
    )

    assert response.document_id == document_id
    assert response.total_chunks == response.indexed_points > 0
    assert response.dimension == provider.dimension
    assert "documents" in store.collections
    assert store.collections["documents"]["distance"] == DistanceMetric.COSINE


@pytest.mark.anyio
async def test_index_document_raises_when_no_chunks(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    document_id = uuid.uuid4()
    (upload_dir / f"{document_id}.txt").write_text("", encoding="utf-8")

    with pytest.raises(IndexingError):
        await index_document(
            document_id=document_id,
            upload_dir=upload_dir,
            provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(),
            collection_name="documents",
            distance=DistanceMetric.COSINE,
            strategy=ChunkingStrategy.CHARACTER,
            chunk_size=100,
            overlap=20,
        )


@pytest.mark.anyio
async def test_points_carry_full_payload(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    document_id = uuid.uuid4()
    (upload_dir / f"{document_id}.txt").write_text(
        "payload check text", encoding="utf-8"
    )

    store = FakeVectorStore()

    await index_document(
        document_id=document_id,
        upload_dir=upload_dir,
        provider=FakeEmbeddingProvider(),
        vector_store=store,
        collection_name="documents",
        distance=DistanceMetric.COSINE,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=100,
        overlap=20,
    )

    points = store.points["documents"]
    assert len(points) == 1
    payload = points[0].payload
    assert payload.document_id == document_id
    assert payload.page_number == 1
    assert payload.chunk_index == 0
    assert "payload check" in payload.content
    assert len(points[0].vector) == 4
