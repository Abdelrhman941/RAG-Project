import hashlib
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core import SourceType
from app.schemas.chunk import Chunk, ChunkRequest


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
        content_hash=hashlib.sha256("Hello world".encode()).hexdigest(),
    )
    assert chunk.source_type == SourceType.TXT


def test_chunk_accepts_parent_fields() -> None:
    parent_id = uuid4()
    chunk = Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="test",
        start_char=0,
        end_char=4,
        char_count=4,
        source_type=SourceType.PDF,
        content_hash="abcd",
        parent_chunk_id=parent_id,
        parent_content="test parent",
    )
    assert chunk.parent_chunk_id == parent_id
    assert chunk.parent_content == "test parent"


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


def test_chunk_request_validates_prompt_overlap() -> None:
    with pytest.raises(
        ValidationError, match="prompt_overlap must be smaller than prompt_chunk_size"
    ):
        ChunkRequest(prompt_chunk_size=500, prompt_overlap=500)


def test_chunk_request_validates_embedding_smaller_than_prompt() -> None:
    with pytest.raises(
        ValidationError,
        match="embedding_chunk_size must be strictly smaller than prompt_chunk_size",
    ):
        ChunkRequest(embedding_chunk_size=1000, prompt_chunk_size=500)
