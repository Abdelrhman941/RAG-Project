import uuid

from fastapi.testclient import TestClient


def _upload_txt(client: TestClient, content: bytes, filename: str = "notes.txt") -> str:
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, content, "text/plain")},
    )
    document_id: str = response.json()["id"]
    return document_id


def test_embed_document_happy_path(client: TestClient) -> None:
    document_id = _upload_txt(client, b"a" * 2500)

    response = client.post(f"/api/v1/documents/{document_id}/embed")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["embedding_model"] == "fake-embedding-model"
    assert body["dimension"] == 4
    assert body["status"] == "completed"


def test_embed_document_total_chunks_matches_chunker_output(
    client: TestClient,
) -> None:
    # DEFAULT_CHUNK_SIZE=1000 / DEFAULT_CHUNK_OVERLAP=200 -> step=800
    # 2500 chars -> chunks at 0, 800, 1600 -> 3 chunks
    document_id = _upload_txt(client, b"a" * 2500)

    response = client.post(f"/api/v1/documents/{document_id}/embed")

    assert response.json()["total_chunks"] == 3


def test_embed_nonexistent_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/api/v1/documents/{uuid.uuid4()}/embed")

    assert response.status_code == 404


def test_embed_invalid_uuid_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/documents/not-a-uuid/embed")

    assert response.status_code == 422


def test_embed_corrupted_pdf_returns_500(client: TestClient) -> None:
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/embed")

    assert response.status_code == 500


def test_embed_response_matches_embedding_response_schema(client: TestClient) -> None:
    document_id = _upload_txt(client, b"hello world")

    response = client.post(f"/api/v1/documents/{document_id}/embed")

    body = response.json()
    assert set(body.keys()) == {
        "document_id",
        "total_chunks",
        "embedding_model",
        "dimension",
        "status",
    }
