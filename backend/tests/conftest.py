import importlib
import io
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.core import get_settings


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")

    get_settings.cache_clear()

    from app import main as main_module

    importlib.reload(main_module)

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
