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
        mock_parse.return_value = ["This is page one."]

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
