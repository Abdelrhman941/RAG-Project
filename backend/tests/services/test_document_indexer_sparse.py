from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, DistanceMetric, SourceType
from app.embedders.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.schemas.chunk import Chunk
from app.schemas.sparse import SparseVector
from app.services.document_chunker import DocumentChunkerService
from app.services.document_embedder import DocumentEmbedderService
from app.services.document_indexer import DocumentIndexerService
from app.vectorstores.base import BaseVectorStore


class MockDenseProvider(BaseEmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "dense"

    @property
    def embedding_dimension(self) -> int:
        return 2

    @property
    def max_sequence_length(self) -> int:
        return 100

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class MockSparseProvider(BaseSparseEmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "sparse"

    def embed_sparse_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    def embed_sparse_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


@pytest.mark.asyncio
async def test_index_document_with_sparse() -> None:
    doc_id = uuid4()
    chunks = [
        Chunk(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            content="hello",
            source_type=SourceType.TXT,
            start_char=0,
            end_char=5,
            char_count=5,
            content_hash="hash",
        )
    ]

    async def mock_chunk_gen(*args: Any, **kwargs: Any) -> AsyncIterator[Chunk]:
        for c in chunks:
            yield c

    from unittest.mock import MagicMock

    mock_chunker = MagicMock(spec=DocumentChunkerService)
    mock_chunker.chunk_document = mock_chunk_gen

    dense_provider = MockDenseProvider()
    sparse_provider = MockSparseProvider()

    mock_embedder = MagicMock(spec=DocumentEmbedderService)
    # The default mock returns an AsyncMock which can be awaited,
    # but let's give it proper return values.
    mock_embedder.embed_chunks = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock(spec=BaseVectorStore)
    vector_store.get_existing_hashes.return_value = frozenset()

    indexer = DocumentIndexerService(
        chunker=mock_chunker,
        embedder=mock_embedder,
        provider=dense_provider,
        vector_store=vector_store,
        sparse_provider=sparse_provider,
    )

    await indexer.index_document(
        document_id=doc_id,
        collection_name="col",
        distance=DistanceMetric.COSINE,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=100,
        embedding_overlap=10,
    )

    vector_store.upsert.assert_called_once()
    upserted = vector_store.upsert.call_args.kwargs["points"]
    assert len(upserted) == 1
    point = upserted[0]

    # Assert point has both dense and sparse
    assert point.vector == [0.1, 0.2]
    assert point.sparse_vector is not None
    assert point.sparse_vector.indices == [1]
    assert point.sparse_vector.values == [1.0]
