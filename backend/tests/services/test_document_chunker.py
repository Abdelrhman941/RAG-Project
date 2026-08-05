from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, SourceType
from app.services.document_chunker import DocumentChunkerService
from app.services.document_parser import DocumentParserService


@pytest.mark.asyncio
async def test_chunk_document_propagates_source_type() -> None:
    doc_id = uuid4()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_parse_cm(*args, **kwargs):
        async def mock_parse_gen():
            for page in ["This is page one."]:
                yield page

        yield (mock_parse_gen(), SourceType.TXT)

    mock_parser = MagicMock(spec=DocumentParserService)
    mock_parser.parse_document = mock_parse_cm

    chunker = DocumentChunkerService(parser=mock_parser, min_chunk_chars=10)

    chunks = [
        c
        async for c in chunker.chunk_document(
            document_id=doc_id,
            strategy=ChunkingStrategy.TOKEN,
            embedding_chunk_size=10,
            embedding_overlap=0,
        )
    ]

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.source_type == SourceType.TXT
        assert hasattr(chunk, "start_char")
        assert hasattr(chunk, "end_char")


@pytest.mark.asyncio
async def test_chunk_document_dedup_and_contiguous_index() -> None:
    """Intra-document duplicate chunks must be dropped; surviving indices must
    form a contiguous sequence starting at 0."""
    doc_id = uuid4()
    # Page texts chosen so each becomes exactly one small chunk when
    # chunk_size is very small. The footer appears on pages 2 and 4.
    pages = [
        "Unique content on page one here.",
        "Boilerplate footer.",
        "Unique content on page three here.",
        "Boilerplate footer.",
    ]

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_parse_cm(*args, **kwargs):
        async def mock_parse_gen():
            for page in pages:
                yield page

        yield (mock_parse_gen(), SourceType.TXT)

    mock_parser = MagicMock(spec=DocumentParserService)
    mock_parser.parse_document = mock_parse_cm

    chunker = DocumentChunkerService(
        parser=mock_parser, min_chunk_chars=10, dedup_similarity_threshold=1.0
    )

    chunks = [
        c
        async for c in chunker.chunk_document(
            document_id=doc_id,
            strategy=ChunkingStrategy.TOKEN,
            embedding_chunk_size=200,
            embedding_overlap=0,
        )
    ]

    # The second identical footer should be dropped.
    assert len(chunks) == 3
    contents = [c.content for c in chunks]
    assert contents.count("Boilerplate footer.") == 1

    # chunk_index must be contiguous: [0, 1, 2]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
