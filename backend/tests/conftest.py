import importlib
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

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
