from collections.abc import Sequence
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.embedders.base import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    BaseSparseEmbeddingProvider,
)
from app.retrieval.models import SearchResult
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


class MockRerankerProvider(BaseRerankerProvider):
    @property
    def model_name(self) -> str:
        return "reranker"

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        return [0.9, 0.1, 0.8]  # Corresponds to text1, text2, text3


@pytest.mark.asyncio
async def test_retrieve_with_reranker() -> None:
    dense_provider = MockDenseProvider()
    sparse_provider = MockSparseProvider()
    reranker = MockRerankerProvider()

    vector_store = AsyncMock(spec=BaseVectorStore)
    vector_store.search.return_value = [
        SearchResult(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.016,
            content="text1",
        ),
        SearchResult(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.015,
            content="text2",
        ),
        SearchResult(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.014,
            content="text3",
        ),
    ]

    results = await retrieve(
        query="test query",
        top_k=2,
        fetch_k=15,
        rerank_min_score=0.3,
        collection_name="col",
        provider=dense_provider,
        vector_store=vector_store,
        sparse_provider=sparse_provider,
        reranker=reranker,
    )

    vector_store.search.assert_called_once()
    kwargs = vector_store.search.call_args.kwargs
    # Because reranker is present, it should fetch fetch_k and disable min_score
    assert kwargs["top_k"] == 15
    assert kwargs["min_score"] is None

    # Output should be top 2 reranked: text1 (0.9) and text3 (0.8).
    # text2 (0.1) is filtered out.
    assert len(results) == 2
    assert results[0].content == "text1"
    assert results[0].rerank_score == 0.9
    assert results[0].score == 0.016  # Preserve original
    assert results[1].content == "text3"
    assert results[1].rerank_score == 0.8
    assert results[1].score == 0.014  # Preserve original
