"""Unit tests for the generation domain models (Sprint 9, Phase 1)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.generation import (
    ChatMessage,
    ChatRole,
    Citation,
    GenerationResult,
    TokenUsage,
)


class TestChatMessage:
    def test_valid_construction(self) -> None:
        msg = ChatMessage(role=ChatRole.SYSTEM, content="hello")
        assert msg.role is ChatRole.SYSTEM
        assert msg.content == "hello"

    def test_is_immutable(self) -> None:
        msg = ChatMessage(role=ChatRole.USER, content="hi")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_empty_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role=ChatRole.USER, content="")


class TestCitation:
    def test_valid_construction(self) -> None:
        cit = Citation(
            document_id=uuid4(),
            chunk_id=uuid4(),
            page_number=3,
            score=0.72,
        )
        assert cit.page_number == 3
        assert cit.score == pytest.approx(0.72)

    def test_is_immutable(self) -> None:
        cit = Citation(
            document_id=uuid4(),
            chunk_id=uuid4(),
            page_number=1,
            score=0.5,
        )
        with pytest.raises(ValidationError):
            cit.score = 0.9  # type: ignore[misc]

    def test_page_number_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation(
                document_id=uuid4(),
                chunk_id=uuid4(),
                page_number=0,
                score=0.5,
            )

    def test_score_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation(
                document_id=uuid4(),
                chunk_id=uuid4(),
                page_number=1,
                score=1.5,
            )


class TestTokenUsage:
    def test_defaults_to_zero(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_negative_tokens_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)


class TestGenerationResult:
    def test_valid_construction_with_defaults(self) -> None:
        result = GenerationResult(answer="42", model="fake-llm")
        assert result.answer == "42"
        assert result.model == "fake-llm"
        assert result.citations == []
        assert result.usage.total_tokens == 0

    def test_is_immutable(self) -> None:
        result = GenerationResult(answer="x", model="m")
        with pytest.raises(ValidationError):
            result.answer = "y"  # type: ignore[misc]

    def test_accepts_citations(self) -> None:
        cit = Citation(
            document_id=uuid4(),
            chunk_id=uuid4(),
            page_number=1,
            score=0.9,
        )
        result = GenerationResult(
            answer="see ref",
            model="fake-llm",
            citations=[cit],
        )
        assert len(result.citations) == 1
        assert result.citations[0].score == pytest.approx(0.9)
