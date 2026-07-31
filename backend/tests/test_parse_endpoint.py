import uuid
from typing import Callable

from fastapi.testclient import TestClient


def test_parse_txt_document_returns_single_page(client: TestClient) -> None:
    content = b"Hello, this is a test document."
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/parse")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["pages"] == [content.decode("utf-8")]


def test_parse_pdf_document_returns_expected_pages(
    client: TestClient, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    pdf_bytes = make_pdf_bytes(["First page", "Second page"])
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/parse")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert len(body["pages"]) == 2
    assert "First page" in body["pages"][0]
    assert "Second page" in body["pages"][1]


def test_parse_md_document_returns_raw_markdown(client: TestClient) -> None:
    content = b"# Title\n\nSome markdown body."
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("readme.md", content, "text/markdown")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/parse")

    assert response.status_code == 200
    assert response.json()["pages"] == [content.decode("utf-8")]


def test_parse_nonexistent_document_returns_404(client: TestClient) -> None:
    random_id = uuid.uuid4()

    response = client.post(f"/api/v1/documents/{random_id}/parse")

    assert response.status_code == 404
    assert str(random_id) in response.json()["detail"]


def test_parse_invalid_uuid_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/documents/not-a-uuid/parse")

    assert response.status_code == 422


def test_parse_corrupted_pdf_returns_500(client: TestClient) -> None:
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/parse")

    assert response.status_code == 500
    assert "Could not read" in response.json()["detail"]


def test_parse_response_matches_parsed_document_schema(client: TestClient) -> None:
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    document_id = upload_response.json()["id"]

    response = client.post(f"/api/v1/documents/{document_id}/parse")

    body = response.json()
    assert set(body.keys()) == {"document_id", "pages"}
    assert isinstance(body["pages"], list)
