"""Prompt building layer (Sprint 9, Phase 1).

Turns a user question + retrieved chunks into a list of `ChatMessage`s
ready to be handed to *any* LLM provider adapter. The builder itself
never imports Groq, OpenAI, FastAPI, or an HTTP client.

Design notes
------------
- Pure function-object style: no I/O, no globals, deterministic output.
- Accepts anything that quacks like `SearchResult` / `RetrievedChunk`
    via a small structural protocol, so callers can pass either the
    retrieval domain model or the public schema without a conversion step.
    This keeps retrieval and generation cleanly decoupled.
- Empty retrieval is a normal case (not an error): the system prompt
    explicitly tells the model to refuse when context is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import ChatMessage, ChatRole

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
# Kept as a module-level constant so it is easy to A/B, snapshot-test,
# and later externalise into a config file without touching callers.
SYSTEM_PROMPT: str = (
    "You are a helpful, precise assistant answering questions strictly "
    "from the provided context. Follow these rules:\n"
    "1. Use ONLY the information contained in the CONTEXT block. Do not "
    "invent facts, do not use outside knowledge.\n"
    "2. If the answer cannot be found in the context, reply exactly: "
    '"I don\'t know based on the provided context."\n'
    "3. Be concise. Prefer direct answers over long explanations.\n"
    "4. When you use a specific piece of context, cite it inline using "
    "the form [chunk N] where N is the 1-based index of the chunk in the "
    "CONTEXT block."
)

# Sentinel used when a chunk has no usable content — keeps the prompt
# well-formed even if an upstream bug ever slips an empty string through.
_EMPTY_CONTENT_PLACEHOLDER = "(empty)"


@runtime_checkable
class RetrievedChunkLike(Protocol):
    """Structural protocol for anything the builder can consume.

    Both `app.retrieval.SearchResult` and
    `app.schemas.RetrievedChunk` satisfy this — no explicit
    inheritance needed, no runtime coupling introduced.
    """

    page_number: int
    score: float
    content: str


class PromptBuilder:
    """Builds chat messages for a RAG generation call.

    Responsibilities:
        - Assemble the system message.
        - Assemble the user message containing (context + question).

    Non-responsibilities (out of scope for Phase 1):
        - Calling an LLM.
        - Knowing HTTP, FastAPI, or Groq.
        - Truncating context to a token budget (future phase).
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        # Allow injection for tests / future config, but default to the
        # module-level template so callers can stay parameter-free.
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty.")
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        query: str,
        chunks: Sequence[RetrievedChunkLike],
    ) -> list[ChatMessage]:
        """Return the ordered list of messages to send to the LLM.

        Parameters
        ----------
        query:
            The user question. Must be non-empty after stripping.
        chunks:
            Retrieved evidence, already ranked by the retrieval layer.
            The builder does NOT reorder or filter these — it trusts the
            caller. An empty sequence is allowed.

        Returns
        -------
        A list containing exactly one system message followed by exactly
        one user message.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty or whitespace-only.")

        user_content = self._compose_user_message(cleaned_query, chunks)

        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt),
            ChatMessage(role=ChatRole.USER, content=user_content),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compose_user_message(
        self,
        query: str,
        chunks: Sequence[RetrievedChunkLike],
    ) -> str:
        context_block = self._format_context(chunks)
        return (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{query}\n\n"
            "Answer the question using ONLY the context above. "
            "Cite chunks inline as [chunk N]."
        )

    @staticmethod
    def _format_context(chunks: Sequence[RetrievedChunkLike]) -> str:
        if not chunks:
            return "(no relevant context was retrieved)"

        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            content = (chunk.content or "").strip() or _EMPTY_CONTENT_PLACEHOLDER
            lines.append(
                f"[chunk {idx}] (page {chunk.page_number}, "
                f"score {chunk.score:.3f})\n{content}"
            )
        return "\n\n".join(lines)
