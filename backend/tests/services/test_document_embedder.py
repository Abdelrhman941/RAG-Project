from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, EmbeddingError
from app.services.document_embedder import embed_document


@pytest.mark.asyncio
async def test_embed_document_rejects_oversized_chunk() -> None:
    mock_provider = MagicMock()
    mock_provider.max_sequence_length = 512

    with pytest.raises(EmbeddingError, match="exceeds model maximum context length"):
        await embed_document(
            document_id=uuid4(),
            upload_dir=MagicMock(),
            provider=mock_provider,
            strategy=ChunkingStrategy.TOKEN,
            embedding_chunk_size=1000,
            embedding_overlap=0,
        )
