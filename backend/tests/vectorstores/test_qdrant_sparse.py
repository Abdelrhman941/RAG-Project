from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from qdrant_client.http import models as qmodels

from app.core import DistanceMetric, SourceType
from app.schemas.sparse import SparseVector
from app.vectorstores.models import PointData, PointPayload
from app.vectorstores.qdrant import QdrantVectorStore


@pytest.fixture
def mock_qdrant() -> Generator[AsyncMock, None, None]:
    with patch("app.vectorstores.qdrant.AsyncQdrantClient", autospec=True) as mock_cls:
        client = mock_cls.return_value
        client.collection_exists.return_value = False
        yield client


@pytest.mark.asyncio
async def test_qdrant_create_collection_with_sparse(mock_qdrant: AsyncMock) -> None:
    store = QdrantVectorStore(host="localhost", port=6333)

    await store.create_collection(
        "test_col", dimension=128, distance=DistanceMetric.COSINE
    )

    mock_qdrant.create_collection.assert_called_once()
    kwargs = mock_qdrant.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "test_col"
    assert "sparse_vectors_config" in kwargs
    assert "sparse" in kwargs["sparse_vectors_config"]
    assert isinstance(
        kwargs["sparse_vectors_config"]["sparse"], qmodels.SparseVectorParams
    )


@pytest.mark.asyncio
async def test_qdrant_upsert_hybrid_vectors(mock_qdrant: AsyncMock) -> None:
    store = QdrantVectorStore(host="localhost", port=6333)

    doc_id = uuid4()
    chunk_id = uuid4()
    payload = PointPayload(
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        page_number=1,
        content="hello",
        source_type=SourceType.TXT,
        start_char=0,
        end_char=5,
        content_hash="hash",
    )
    sv = SparseVector(indices=[1, 5], values=[0.1, 0.9])

    point = PointData(id=chunk_id, vector=[0.1, 0.2], payload=payload, sparse_vector=sv)

    await store.upsert("test_col", [point])

    mock_qdrant.upsert.assert_called_once()
    upserted_points = mock_qdrant.upsert.call_args.kwargs["points"]
    assert len(upserted_points) == 1

    qpoint = upserted_points[0]
    # Qdrant expects vector to be a dict if sparse is used
    assert isinstance(qpoint.vector, dict)
    assert qpoint.vector[""] == [0.1, 0.2]  # dense vector
    assert "sparse" in qpoint.vector
    assert isinstance(qpoint.vector["sparse"], qmodels.SparseVector)
    assert qpoint.vector["sparse"].indices == [1, 5]
    assert qpoint.vector["sparse"].values == [0.1, 0.9]


@pytest.mark.asyncio
async def test_qdrant_search_hybrid_rrf(mock_qdrant: AsyncMock) -> None:
    store = QdrantVectorStore(host="localhost", port=6333)
    mock_qdrant.query_points.return_value = MagicMock(points=[])

    sv = SparseVector(indices=[1, 5], values=[0.1, 0.9])

    # We should use query_points with prefetch for hybrid search
    await store.search("test_col", vector=[0.1, 0.2], top_k=5, sparse_vector=sv)

    mock_qdrant.query_points.assert_called_once()
    kwargs = mock_qdrant.query_points.call_args.kwargs

    # Assert query_points is used instead of search
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["limit"] == 5

    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2

    # One prefetch for dense, one for sparse
    assert any(isinstance(p.query, list) and not p.using for p in prefetch)  # dense
    assert any(
        isinstance(p.query, qmodels.SparseVector) and p.using == "sparse"
        for p in prefetch
    )  # sparse

    # Query itself should be FusionQuery
    assert kwargs["query"] == qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)
