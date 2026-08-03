"""
Factory for creating LLM providers.

Design:
- create_llm_provider(settings): pure factory function.
- get_llm_provider(): cached singleton used by the application.
"""

from functools import lru_cache

from ..core import (
    LLMProviderName,
    Settings,
    UnknownProviderError,
    get_settings,
)
from .base import BaseLLMProvider
from .groq import GroqProvider


def create_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Create an LLM provider from application settings."""
    if settings.LLM_PROVIDER is LLMProviderName.GROQ:
        return GroqProvider(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    raise UnknownProviderError(settings.LLM_PROVIDER)


@lru_cache
def get_llm_provider() -> BaseLLMProvider:
    """Return the application-wide LLM provider singleton."""
    return create_llm_provider(get_settings())
