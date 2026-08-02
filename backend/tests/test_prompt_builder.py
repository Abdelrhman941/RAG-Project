"""Unit tests for the PromptBuilder (Sprint 9, Phase 1)."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from app.generation import ChatRole, PromptBuilder
from app.generation.prompt_builder import SYSTEM_PROMPT, RetrievedChunkLike
from app.retrieval import SearchResult


def _make_search_result(
    *,
    page_number: int = 1,
    score: float = 0.5,
    content: str = "sample content",
) -> RetrievedChunkLike:
    result = SearchResult(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=page_number,
        score=score,
        content=content,
    )
    return cast(RetrievedChunkLike, result)


class TestPromptBuilderStructure:
    def test_returns_exactly_two_messages(self) -> None:
        builder = PromptBuilder()
        messages = builder.build("what is x?", [_make_search_result()])
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        builder = PromptBuilder()
        messages = builder.build("q", [_make_search_result()])
        assert messages[0].role is ChatRole.SYSTEM
        assert messages[0].content == SYSTEM_PROMPT

    def test_second_message_is_user(self) -> None:
        builder = PromptBuilder()
        messages = builder.build("q", [_make_search_result()])
        assert messages[1].role is ChatRole.USER


class TestPromptBuilderContent:
    def test_user_message_includes_the_question(self) -> None:
        builder = PromptBuilder()
        messages = builder.build(
            "What is the capital of France?", [_make_search_result()]
        )
        assert "What is the capital of France?" in messages[1].content

    def test_user_message_includes_retrieved_context(self) -> None:
        builder = PromptBuilder()
        chunk = _make_search_result(content="Paris is the capital of France.")
        messages = builder.build("q?", [chunk])
        assert "Paris is the capital of France." in messages[1].content

    def test_context_is_indexed_starting_at_one(self) -> None:
        builder = PromptBuilder()
        chunks = [
            _make_search_result(content="alpha"),
            _make_search_result(content="beta"),
        ]
        user_content = builder.build("q?", chunks)[1].content
        assert "[chunk 1]" in user_content
        assert "[chunk 2]" in user_content
        # Order preserved (retrieval is authoritative on ranking).
        assert user_content.index("alpha") < user_content.index("beta")

    def test_context_shows_page_and_score(self) -> None:
        builder = PromptBuilder()
        chunk = _make_search_result(page_number=7, score=0.876, content="x")
        user_content = builder.build("q?", [chunk])[1].content
        assert "page 7" in user_content
        assert "0.876" in user_content

    def test_empty_content_falls_back_to_placeholder(self) -> None:
        # The domain model rejects empty content, so simulate an
        # upstream-tampered object with a stub.
        class _Stub:
            page_number = 1
            score = 0.5
            content = "   "

        builder = PromptBuilder()
        user_content = builder.build("q?", [_Stub()]).__getitem__(1).content
        assert "(empty)" in user_content


class TestPromptBuilderEmptyRetrieval:
    def test_empty_chunks_are_allowed(self) -> None:
        builder = PromptBuilder()
        messages = builder.build("q?", [])
        assert len(messages) == 2
        assert messages[0].role is ChatRole.SYSTEM
        assert messages[1].role is ChatRole.USER

    def test_empty_chunks_produce_explicit_no_context_marker(self) -> None:
        builder = PromptBuilder()
        user_content = builder.build("q?", [])[1].content
        assert "no relevant context" in user_content.lower()

    def test_empty_chunks_still_include_the_question(self) -> None:
        builder = PromptBuilder()
        user_content = builder.build("original question", [])[1].content
        assert "original question" in user_content


class TestPromptBuilderValidation:
    def test_empty_query_is_rejected(self) -> None:
        builder = PromptBuilder()
        with pytest.raises(ValueError):
            builder.build("", [])

    def test_whitespace_only_query_is_rejected(self) -> None:
        builder = PromptBuilder()
        with pytest.raises(ValueError):
            builder.build("   \n\t  ", [])

    def test_custom_system_prompt_is_used(self) -> None:
        builder = PromptBuilder(system_prompt="You are terse.")
        messages = builder.build("q?", [])
        assert messages[0].content == "You are terse."

    def test_empty_custom_system_prompt_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PromptBuilder(system_prompt="   ")


class TestPromptBuilderDecoupling:
    """PromptBuilder must not import Groq, OpenAI, FastAPI, or HTTP libs."""

    def test_builder_module_has_no_llm_or_http_imports(self) -> None:
        """Guardrail: PromptBuilder must stay provider-agnostic.

        We only inspect *import statements* (not docstrings / comments),
        so mentioning "Groq" in a comment is fine — actually importing
        it is not.
        """
        import ast

        import app.generation.prompt_builder as pb_mod

        source_path = pb_mod.__file__
        assert source_path is not None
        with open(source_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module)

        forbidden = ("groq", "openai", "fastapi", "httpx", "requests")
        offenders = [
            name
            for name in imported
            if any(name == f or name.startswith(f + ".") for f in forbidden)
        ]
        assert not offenders, f"Forbidden imports found: {offenders}"

    def test_accepts_retrieved_chunk_schema_too(self) -> None:
        """Structural typing: RetrievedChunk should also work."""
        from typing import cast

        from app.schemas import RetrievedChunk

        chunk = RetrievedChunk(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=2,
            score=0.42,
            content="from schema",
        )

        casted_chunk = cast(RetrievedChunkLike, chunk)

        builder = PromptBuilder()
        user_content = builder.build("q?", [casted_chunk])[1].content
        assert "from schema" in user_content
        assert "page 2" in user_content
