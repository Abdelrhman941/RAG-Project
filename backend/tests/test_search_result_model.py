"""Unit tests for the `SearchResult` domain model (Sprint 8)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.retrieval import SearchResult


class TestSearchResult:
    def test_valid_construction(self) -> None:
        result = SearchResult(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.87,
            content="hello",
        )
        assert result.score == pytest.approx(0.87)
        assert result.content == "hello"

    def test_is_immutable(self) -> None:
        result = SearchResult(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.5,
            content="x",
        )
        with pytest.raises(ValidationError):
            result.score = 0.9  # type: ignore[misc]

    def test_negative_chunk_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(
                document_id=uuid4(),
                chunk_id=uuid4(),
                chunk_index=-1,
                page_number=1,
                score=0.5,
                content="x",
            )

    def test_page_number_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(
                document_id=uuid4(),
                chunk_id=uuid4(),
                chunk_index=0,
                page_number=0,
                score=0.5,
                content="x",
            )

    def test_score_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(
                document_id=uuid4(),
                chunk_id=uuid4(),
                chunk_index=0,
                page_number=1,
                score=1.5,
                content="x",
            )
