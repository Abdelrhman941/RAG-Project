"""
Groq adapter — the ONLY file in the codebase allowed to import the Groq SDK.
"""

from groq import AsyncGroq
from groq import GroqError as GroqSDKError

from ..core import LLMProviderError
from .base import BaseLLMProvider, ChatMessage


class GroqProvider(BaseLLMProvider):
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
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature
                if temperature is not None
                else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            )
        except GroqSDKError as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc
        except Exception as exc:  # network errors, timeouts, etc.
            raise LLMProviderError(f"Unexpected Groq client error: {exc}") from exc

        if not response.choices:
            raise LLMProviderError("Groq returned no choices")

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("Groq returned an empty message")

        return content
