"""Unit tests for `QdrantVectorStore._to_search_result` (adapter mapping).

We test the mapping in isolation, without spinning up a real Qdrant
instance, because the mapping is where SDK types are supposed to stop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.core import RetrievalError
from app.retrieval import SearchResult
from app.vectorstores.qdrant import QdrantVectorStore


def _fake_scored_point(
    payload: dict[str, Any] | None, score: float = 0.87
) -> SimpleNamespace:
    """Minimal stand-in for `qdrant_client.http.models.ScoredPoint`."""
    return SimpleNamespace(id="fake-id", score=score, payload=payload)


class TestScoredPointMapping:
    def test_valid_payload_maps_to_search_result(self) -> None:
        document_id = uuid4()
        chunk_id = uuid4()
        point = _fake_scored_point(
            payload={
                "document_id": str(document_id),
                "chunk_id": str(chunk_id),
                "chunk_index": 3,
                "page_number": 2,
                "content": "DBSCAN is a density-based clustering algorithm.",
            },
            score=0.91,
        )

        result = QdrantVectorStore._to_search_result(point)  # type: ignore[arg-type]

        assert isinstance(result, SearchResult)
        assert result.document_id == document_id
        assert result.chunk_id == chunk_id
        assert result.chunk_index == 3
        assert result.page_number == 2
        assert result.score == pytest.approx(0.91)
        assert result.content.startswith("DBSCAN")

    def test_missing_payload_field_raises_retrieval_error(self) -> None:
        point = _fake_scored_point(
            payload={
                "document_id": str(uuid4()),
                # chunk_id intentionally missing
                "chunk_index": 0,
                "page_number": 1,
                "content": "x",
            }
        )
        with pytest.raises(RetrievalError):
            QdrantVectorStore._to_search_result(point)  # type: ignore[arg-type]

    def test_empty_payload_raises_retrieval_error(self) -> None:
        point = _fake_scored_point(payload=None)
        with pytest.raises(RetrievalError):
            QdrantVectorStore._to_search_result(point)  # type: ignore[arg-type]

    def test_malformed_uuid_raises_retrieval_error(self) -> None:
        point = _fake_scored_point(
            payload={
                "document_id": "not-a-uuid",
                "chunk_id": str(uuid4()),
                "chunk_index": 0,
                "page_number": 1,
                "content": "x",
            }
        )
        with pytest.raises(RetrievalError):
            QdrantVectorStore._to_search_result(point)  # type: ignore[arg-type]
