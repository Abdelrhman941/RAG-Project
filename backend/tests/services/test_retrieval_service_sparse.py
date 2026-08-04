from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from app.embedders.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.schemas.sparse import SparseVector
from app.services.retrieval_service import retrieve
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
        return []

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class MockSparseProvider(BaseSparseEmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "sparse"

    def embed_sparse_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return []

    def embed_sparse_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


@pytest.mark.asyncio
async def test_retrieve_with_sparse() -> None:
    dense_provider = MockDenseProvider()
    sparse_provider = MockSparseProvider()
    vector_store = AsyncMock(spec=BaseVectorStore)
    vector_store.search.return_value = []

    await retrieve(
        query="test query",
        top_k=5,
        collection_name="col",
        provider=dense_provider,
        vector_store=vector_store,
        sparse_provider=sparse_provider,
    )

    vector_store.search.assert_called_once()
    kwargs = vector_store.search.call_args.kwargs
    assert kwargs["vector"] == [0.1, 0.2]

    assert "sparse_vector" in kwargs
    sv = kwargs["sparse_vector"]
    assert sv is not None
    assert sv.indices == [1]
    assert sv.values == [1.0]
