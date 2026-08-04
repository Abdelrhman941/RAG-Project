from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import ChunkingStrategy, DistanceMetric, SourceType
from app.embedders.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.schemas.chunk import Chunk
from app.schemas.sparse import SparseVector
from app.services.document_indexer import index_document
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
@patch("app.services.document_indexer.chunk_document")
async def test_index_document_with_sparse(
    mock_chunk: MagicMock
) -> None:
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
    mock_chunk.return_value = chunks

    dense_provider = MockDenseProvider()
    sparse_provider = MockSparseProvider()
    vector_store = AsyncMock(spec=BaseVectorStore)
    vector_store.get_existing_hashes.return_value = frozenset()

    await index_document(
        document_id=doc_id,
        upload_dir=Path("/tmp"),
        collection_name="col",
        distance=DistanceMetric.COSINE,
        provider=dense_provider,
        vector_store=vector_store,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=100,
        embedding_overlap=10,
        sparse_provider=sparse_provider,
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
