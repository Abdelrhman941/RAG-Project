"""
Groq adapter.

This is the only module allowed to import the Groq SDK.

Responsibilities:
- Translate ProviderChatMessage -> Groq request format.
- Call the Groq API.
- Translate Groq SDK exceptions into project exceptions.
- Return plain generated text.
"""

from groq import AsyncGroq
from groq import GroqError as GroqSDKError
from groq.resources.chat import AsyncChat

from ..core import LLMProviderError
from .base import BaseLLMProvider, ProviderChatMessage


class GroqProvider(BaseLLMProvider):
    """LLM provider backed by the Groq API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[ProviderChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion using Groq."""
        try:
            chat: AsyncChat = self._client.chat  # cached_property; annotate for Pylance
            response = await chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=(self._temperature if temperature is None else temperature),
                max_tokens=(self._max_tokens if max_tokens is None else max_tokens),
            )
        except GroqSDKError as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc
        except Exception as exc:
            raise LLMProviderError(f"Unexpected Groq client error: {exc}") from exc

        if not response.choices:
            raise LLMProviderError("Groq returned no choices.")

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("Groq returned an empty response.")

        return content
