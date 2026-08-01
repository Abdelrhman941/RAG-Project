import uuid
from collections.abc import Callable

from fastapi.testclient import TestClient


def _upload_txt(client: TestClient, content: bytes, filename: str = "notes.txt") -> str:
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, content, "text/plain")},
    )
    document_id: str = response.json()["id"]
    return document_id


def test_chunk_document_with_defaults_returns_201(client: TestClient) -> None:
    document_id = _upload_txt(client, b"a" * 2500)

    response = client.post(f"/api/v1/documents/{document_id}/chunks")

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == document_id
    assert body["total_chunks"] == len(body["chunks"])
    assert body["total_chunks"] > 0


def test_chunk_document_uses_settings_defaults_when_body_omitted(
    client: TestClient,
) -> None:
    # DEFAULT_CHUNK_SIZE=1000 / DEFAULT_CHUNK_OVERLAP=200 -> step=800
    # 2500 chars -> chunks starting at 0, 800, 1600 -> 3 chunks
    document_id = _upload_txt(client, b"a" * 2500)

    response = client.post(f"/api/v1/documents/{document_id}/chunks")

    assert response.status_code == 201
    assert response.json()["total_chunks"] == 3


def test_chunk_document_respects_custom_chunk_size_and_overlap(
    client: TestClient,
) -> None:
    document_id = _upload_txt(client, b"a" * 2500)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 500, "overlap": 0},
    )

    assert response.status_code == 201
    assert response.json()["total_chunks"] == 5


def test_chunk_response_matches_chunk_response_schema(client: TestClient) -> None:
    document_id = _upload_txt(client, b"hello world")

    response = client.post(f"/api/v1/documents/{document_id}/chunks")

    body = response.json()
    assert set(body.keys()) == {"document_id", "total_chunks", "chunks"}
    chunk = body["chunks"][0]
    assert set(chunk.keys()) == {
        "chunk_id",
        "document_id",
        "chunk_index",
        "page_number",
        "content",
        "start_char",
        "end_char",
        "char_count",
    }
    assert chunk["document_id"] == document_id
    assert chunk["page_number"] == 1
    assert chunk["chunk_index"] == 0


def test_chunk_nonexistent_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/api/v1/documents/{uuid.uuid4()}/chunks")

    assert response.status_code == 404


def test_chunk_invalid_uuid_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/documents/not-a-uuid/chunks")

    assert response.status_code == 422


def test_chunk_overlap_greater_than_chunk_size_returns_422(
    client: TestClient,
) -> None:
    document_id = _upload_txt(client, b"a" * 1000)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 500, "overlap": 800},
    )

    assert response.status_code == 422


def test_chunk_overlap_equal_to_chunk_size_returns_422(client: TestClient) -> None:
    document_id = _upload_txt(client, b"a" * 1000)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 500, "overlap": 500},
    )

    assert response.status_code == 422


def test_chunk_negative_chunk_size_returns_422(client: TestClient) -> None:
    document_id = _upload_txt(client, b"a" * 1000)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": -10},
    )

    assert response.status_code == 422


def test_chunk_negative_overlap_returns_422(client: TestClient) -> None:
    document_id = _upload_txt(client, b"a" * 1000)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"overlap": -5},
    )

    assert response.status_code == 422


def test_chunk_overlap_only_exceeding_default_chunk_size_returns_422(
    client: TestClient,
) -> None:
    # DEFAULT_CHUNK_SIZE=1000; only overlap is supplied and exceeds it.
    # Pydantic can't catch this (chunk_size is None at validation time),
    # so this exercises the service-level resolved-value check.
    document_id = _upload_txt(client, b"a" * 1000)

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"overlap": 5000},
    )

    assert response.status_code == 422


def test_chunk_pdf_document_preserves_page_numbers(
    client: TestClient, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    pdf_bytes = make_pdf_bytes(["Page one text", "Page two text"])
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(
        f"/api/v1/documents/{document_id}/chunks",
        json={"chunk_size": 1000, "overlap": 0},
    )

    assert response.status_code == 201
    page_numbers = {c["page_number"] for c in response.json()["chunks"]}
    assert page_numbers == {1, 2}


def test_chunk_corrupted_pdf_returns_500(client: TestClient) -> None:
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/chunks")

    assert response.status_code == 500
