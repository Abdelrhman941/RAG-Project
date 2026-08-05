from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.core import SourceType
from app.services.document_parser import DocumentParserService


@pytest.mark.asyncio
async def test_parse_document_returns_text_and_source_type() -> None:
    doc_id = uuid4()
    with (
        patch(
            "app.services.document_parser.DocumentParserService._find_document_path"
        ) as mock_find,
        patch("app.services.document_parser.get_parser") as mock_get_parser,
    ):
        mock_path = Path(f"/tmp/{doc_id}.pdf")
        mock_find.return_value = mock_path

        mock_parser_instance = MagicMock()

        def mock_parse_gen():
            yield "Page 1 text"

        mock_parser_instance.parse.return_value = mock_parse_gen()
        mock_get_parser.return_value = mock_parser_instance

        service = DocumentParserService(upload_dir=Path("/tmp"))
        async with service.parse_document(document_id=doc_id) as (
            pages_gen,
            source_type,
        ):
            pages = [p async for p in pages_gen]
            assert pages == ["Page 1 text"]
            assert source_type == SourceType.PDF
