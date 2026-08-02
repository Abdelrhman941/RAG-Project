"""
Structural contracts (Protocols) the GenerationService depends on.

Fields here mirror the ACTUAL retrieval chunk output from Sprint 7/8
(document_id, chunk_id, chunk_index, page_number, score, content).
No `id` / `text` / `source` placeholders — this matches the real payload.
"""

from typing import Protocol, Sequence, runtime_checkable

from pydantic import UUID4

from ..llms import ChatMessage


@runtime_checkable
class RetrievedChunk(Protocol):
    document_id: UUID4
    chunk_id: UUID4
    chunk_index: int
    page_number: int
    score: float
    content: str


class RetrievalServicePort(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> Sequence[RetrievedChunk]:
        """Return top_k chunks for the query, best-first. No reordering downstream."""
        ...


class PromptBuilderPort(Protocol):
    def build(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> list[ChatMessage]:
        """Build the chat messages to send to the LLM provider."""
        ...
