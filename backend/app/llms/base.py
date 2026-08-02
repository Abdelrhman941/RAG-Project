"""
Abstract LLM provider interface.

Rule: the rest of the app (services, use-cases) depends ONLY on this
interface. No Groq SDK type ever leaks outside `llms/groq.py`.
"""

from abc import ABC, abstractmethod
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    """Provider-agnostic chat message shape."""

    role: Literal["system", "user", "assistant"]
    content: str


class BaseLLMProvider(ABC):
    """Contract every LLM provider must implement."""

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Take a list of chat messages and return the generated answer
        as plain text.

        Implementations must:
        - never return None / raise on empty content -> wrap in LLMProviderError
        - translate SDK-specific exceptions into app-level LLM exceptions
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the underlying model identifier (for logging/telemetry)."""
        raise NotImplementedError
