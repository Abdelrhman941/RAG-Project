import hashlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, DistanceMetric, SourceType
from app.schemas import Chunk
from app.services.document_chunker import DocumentChunkerService
from app.services.document_embedder import DocumentEmbedderService
from app.services.document_indexer import DocumentIndexerService, _build_points


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=index,
        page_number=1,
        content=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        source_type=SourceType.TXT,
        content_hash=_sha256(text),
    )


def test_build_points_propagates_content_hash() -> None:
    chunk = _make_chunk("Hello world.")
    points = _build_points([chunk], [[0.1] * 3])
    assert len(points) == 1
    assert points[0].payload.content_hash == chunk.content_hash


def _make_indexer(
    chunks: list[Chunk],
    mock_store: AsyncMock,
    mock_provider: MagicMock,
    sparse_provider: Any = None,
) -> DocumentIndexerService:
    async def mock_chunk_gen(*args: Any, **kwargs: Any) -> AsyncIterator[Chunk]:
        for c in chunks:
            yield c

    mock_chunker = MagicMock(spec=DocumentChunkerService)
    mock_chunker.chunk_document = mock_chunk_gen

    mock_embedder = MagicMock(spec=DocumentEmbedderService)

    return DocumentIndexerService(
        chunker=mock_chunker,
        embedder=mock_embedder,
        provider=mock_provider,
        vector_store=mock_store,
        sparse_provider=sparse_provider,
    )


@pytest.mark.asyncio
async def test_index_document_skips_already_indexed_chunks() -> None:
    """Chunks already in the store must not be embedded or upserted."""
    existing_text = "Already indexed content that is already stored."
    new_text = "Brand new content that is completely unique and fresh."

    doc_id = uuid4()
    mock_store = AsyncMock()
    mock_store.get_existing_hashes = AsyncMock(
        return_value=frozenset({_sha256(existing_text)})
    )
    mock_store.create_collection = AsyncMock()
    mock_store.delete_by_document = AsyncMock()
    mock_store.upsert = AsyncMock(return_value=1)

    mock_provider = MagicMock()
    mock_provider.embedding_dimension = 3
    mock_provider.model_name = "test-model"
    mock_provider.max_sequence_length = 512

    chunks = [_make_chunk(existing_text, 0), _make_chunk(new_text, 1)]
    indexer = _make_indexer(chunks, mock_store, mock_provider)

    mock_embed = AsyncMock(return_value=[[0.1] * 3])
    indexer._embedder.embed_chunks = mock_embed

    await indexer.index_document(
        document_id=doc_id,
        collection_name="test",
        distance=DistanceMetric.COSINE,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=200,
        embedding_overlap=0,
    )

    # embed_chunks called with only the 1 new chunk
    embedded_chunks = mock_embed.call_args[0][0]
    assert len(embedded_chunks) == 1
    assert embedded_chunks[0].content == new_text

    # upsert called with only 1 new point
    upserted_points = mock_store.upsert.call_args[1]["points"]
    assert len(upserted_points) == 1
    assert upserted_points[0].payload.content_hash == _sha256(new_text)


@pytest.mark.asyncio
async def test_index_document_all_chunks_new_embeds_all() -> None:
    """When no hashes exist in the store, all chunks must be embedded and upserted."""
    doc_id = uuid4()
    chunks = [_make_chunk("First chunk.", 0), _make_chunk("Second chunk.", 1)]

    mock_store = AsyncMock()
    mock_store.get_existing_hashes = AsyncMock(return_value=frozenset())
    mock_store.create_collection = AsyncMock()
    mock_store.delete_by_document = AsyncMock()
    mock_store.upsert = AsyncMock(return_value=2)

    mock_provider = MagicMock()
    mock_provider.embedding_dimension = 3
    mock_provider.model_name = "test-model"
    mock_provider.max_sequence_length = 512

    indexer = _make_indexer(chunks, mock_store, mock_provider)

    mock_embed = AsyncMock(return_value=[[0.1] * 3, [0.2] * 3])
    indexer._embedder.embed_chunks = mock_embed

    response = await indexer.index_document(
        document_id=doc_id,
        collection_name="test",
        distance=DistanceMetric.COSINE,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=200,
        embedding_overlap=0,
    )

    assert len(mock_embed.call_args[0][0]) == 2
    assert response.total_chunks == 2


@pytest.mark.asyncio
async def test_index_document_rejects_oversized_chunk() -> None:
    from app.core import IndexingError

    mock_provider = MagicMock()
    mock_provider.max_sequence_length = 512

    mock_chunker = MagicMock(spec=DocumentChunkerService)
    mock_embedder = MagicMock(spec=DocumentEmbedderService)
    indexer = DocumentIndexerService(
        chunker=mock_chunker,
        embedder=mock_embedder,
        provider=mock_provider,
        vector_store=MagicMock(),
    )

    with pytest.raises(IndexingError, match="exceeds model"):
        await indexer.index_document(
            document_id=uuid4(),
            collection_name="test",
            distance=DistanceMetric.COSINE,
            strategy=ChunkingStrategy.TOKEN,
            embedding_chunk_size=1000,
            embedding_overlap=0,
        )
