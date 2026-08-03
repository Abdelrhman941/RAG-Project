"""
Protocols that define the contracts required by the generation layer.

The generation service depends only on these abstractions rather than
concrete retrieval or prompt-building implementations.
"""

from typing import Protocol, Sequence

from ..retrieval import SearchResult
from .models import ChatMessage


class RetrievalServicePort(Protocol):
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Sequence[SearchResult]:
        """Retrieve the most relevant chunks for a query."""
        ...


class PromptBuilderPort(Protocol):
    def build(
        self,
        question: str,
        chunks: Sequence[SearchResult],
    ) -> list[ChatMessage]:
        """Build provider-agnostic chat messages from retrieved chunks."""
        ...
