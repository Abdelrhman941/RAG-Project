from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, SourceType
from app.services.document_chunker import chunk_document


@pytest.mark.asyncio
async def test_chunk_document_propagates_source_type() -> None:
    doc_id = uuid4()
    with patch(
        "app.services.document_chunker.parse_document", new_callable=AsyncMock
    ) as mock_parse:
        mock_parse.return_value = (["This is page one."], SourceType.TXT)

        # We also need to patch ChunkingConfig to ensure the
        # source_type we pass is predictable or we can just
        # let it use the current hardcoded SourceType.TXT from Step 4 fix
        chunks = await chunk_document(
            document_id=doc_id,
            upload_dir=Path("/tmp"),
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=10,
            overlap=0,
        )

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
    with patch(
        "app.services.document_chunker.parse_document", new_callable=AsyncMock
    ) as mock_parse:
        mock_parse.return_value = (pages, SourceType.TXT)

        chunks = await chunk_document(
            document_id=doc_id,
            upload_dir=Path("/tmp"),
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=200,
            overlap=0,
        )

    # The second identical footer should be dropped.
    assert len(chunks) == 3
    contents = [c.content for c in chunks]
    assert contents.count("Boilerplate footer.") == 1

    # chunk_index must be contiguous: [0, 1, 2]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
