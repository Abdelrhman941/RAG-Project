from pathlib import Path

from fastapi.testclient import TestClient


def test_upload_happy_path(client: TestClient, tmp_path: Path) -> None:
    content = b"Hello, this is a test document."
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", content, "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()

    assert body["original_filename"] == "notes.txt"
    assert body["extension"] == ".txt"
    assert body["size_bytes"] == len(content)
    assert body["status"] == "uploaded"
    assert body["filename"].endswith(".txt")

    saved_files = list((tmp_path / "uploads").iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == content


def test_upload_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("virus.exe", b"binary-content", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert "Empty files are not allowed" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_upload_file_too_large(client: TestClient, tmp_path: Path) -> None:
    # MAX_FILE_SIZE_MB=1 set in the client fixture
    oversized_content = b"x" * (2 * 1024 * 1024)  # 2 MiB

    response = client.post(
        "/api/v1/documents",
        files={"file": ("big.txt", oversized_content, "text/plain")},
    )

    assert response.status_code == 413
    assert "Max size is 1 MB" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []
