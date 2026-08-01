"""Unit tests for the `RetrievalRequest` schema (validation rules)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core import get_settings
from app.schemas import RetrievalRequest

settings = get_settings()


class TestRetrievalRequestValidation:
    def test_query_is_stripped_of_surrounding_whitespace(self) -> None:
        request = RetrievalRequest(query="   What is DBSCAN?   ")
        assert request.query == "What is DBSCAN?"

    def test_top_k_defaults_to_settings(self) -> None:
        request = RetrievalRequest(query="hello")
        assert request.top_k == settings.DEFAULT_TOP_K

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="")

    def test_whitespace_only_query_is_rejected(self) -> None:
        # After stripping, the value collapses to "" which fails min_length=1.
        with pytest.raises(ValidationError):
            RetrievalRequest(query="     ")

    def test_top_k_below_minimum_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="hello", top_k=0)

    def test_top_k_above_maximum_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest(query="hello", top_k=settings.MAX_TOP_K + 1)

    def test_top_k_at_maximum_is_accepted(self) -> None:
        request = RetrievalRequest(query="hello", top_k=settings.MAX_TOP_K)
        assert request.top_k == settings.MAX_TOP_K

    def test_missing_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalRequest()  # type: ignore[call-arg]
