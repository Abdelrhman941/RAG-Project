import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeVectorStore


def _upload_txt(client: TestClient, content: str = "hello world") -> str:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("sample.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def test_index_endpoint_returns_201_and_persists_points(
    client: TestClient, fake_vector_store: FakeVectorStore
) -> None:
    document_id = _upload_txt(client, "DBSCAN clustering is nice." * 50)

    response = client.post(f"/api/v1/documents/{document_id}/index")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document_id"] == document_id
    assert body["collection_name"] == "documents"
    assert body["total_chunks"] == body["indexed_points"] > 0
    assert body["dimension"] == 4
    assert body["embedding_model"] == "fake-embedding-model"
    assert body["status"] == "indexed"

    # And they actually landed in the store.
    stored = fake_vector_store.points["documents"]
    assert len(stored) == body["indexed_points"]
    assert all(str(p.payload.document_id) == document_id for p in stored)


def test_index_creates_collection_with_expected_dimension(
    client: TestClient, fake_vector_store: FakeVectorStore
) -> None:
    document_id = _upload_txt(client)

    client.post(f"/api/v1/documents/{document_id}/index")

    assert "documents" in fake_vector_store.collections
    assert fake_vector_store.collections["documents"]["dimension"] == 4


def test_reindexing_deletes_previous_points(
    client: TestClient, fake_vector_store: FakeVectorStore
) -> None:
    document_id = _upload_txt(client, "reindex me")

    first = client.post(f"/api/v1/documents/{document_id}/index").json()
    second = client.post(f"/api/v1/documents/{document_id}/index").json()

    # The delete-by-document hook was called before the second upsert.
    assert (
        "documents",
        document_id,
    ) in fake_vector_store.deleted_documents

    # And the final state contains only the second run's points.
    assert len(fake_vector_store.points["documents"]) == second["indexed_points"]
    assert first["indexed_points"] == second["indexed_points"]


def test_index_unknown_document_returns_404(client: TestClient) -> None:
    ghost = uuid.uuid4()
    response = client.post(f"/api/v1/documents/{ghost}/index")
    assert response.status_code == 404


def test_index_invalid_uuid_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/documents/not-a-uuid/index")
    assert response.status_code == 422


def test_index_empty_document_bubbles_as_500(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document whose parsed text yields zero chunks should not silently succeed."""
    # Directly drop an empty .txt into the upload dir so we bypass the
    # upload endpoint's empty-file guard.
    from app.core import get_settings

    upload_dir = get_settings().UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = uuid.uuid4()
    (upload_dir / f"{document_id}.txt").write_text("", encoding="utf-8")

    response = client.post(f"/api/v1/documents/{document_id}/index")
    assert response.status_code == 500
    assert "no chunks" in response.json()["detail"].lower()
