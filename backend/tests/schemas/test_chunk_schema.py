from uuid import uuid4

import pytest
from pydantic import ValidationError
from app.core import SourceType
from app.schemas.chunk import Chunk


def test_chunk_accepts_source_type() -> None:
    chunk = Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="Hello world",
        start_char=0,
        end_char=11,
        char_count=11,
        source_type=SourceType.TXT,
    )
    assert chunk.source_type == SourceType.TXT


def test_chunk_rejects_missing_source_type() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            page_number=1,
            content="hello",
            start_char=0,
            end_char=5,
            char_count=5,
        )  # type: ignore
