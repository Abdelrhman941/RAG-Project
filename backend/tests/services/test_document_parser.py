from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.core import SourceType
from app.services.document_parser import parse_document


@pytest.mark.asyncio
async def test_parse_document_returns_text_and_source_type() -> None:
    doc_id = uuid4()
    with patch("app.services.document_parser.find_document_path") as mock_find, \
         patch("app.services.document_parser.get_parser") as mock_get_parser:
        mock_path = Path(f"/tmp/{doc_id}.pdf")
        mock_find.return_value = mock_path
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ["Page 1 text"]
        mock_get_parser.return_value = mock_parser_instance
        
        pages, source_type = await parse_document(doc_id, Path("/tmp"))
        
        assert pages == ["Page 1 text"]
        assert source_type == SourceType.PDF
