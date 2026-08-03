"""
Factory for LLM providers.

Two layers on purpose:
- `create_llm_provider(settings)` -> pure function, easy to unit test,
    no caching, no FastAPI coupling.
- `get_llm_provider()` -> FastAPI dependency, cached with lru_cache so
    we don't build a new Groq client (and TCP connection pool) on every request.
    This is the fix for the "new provider instance per request" bug pattern.
"""

from functools import lru_cache

from ..core import Settings, UnknownProviderError, get_settings
from .base import BaseLLMProvider
from .groq import GroqProvider

settings = get_settings()


def create_llm_provider(cfg: Settings) -> BaseLLMProvider:
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "groq":
        return GroqProvider(
            api_key=cfg.GROQ_API_KEY,
            model=cfg.GROQ_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
        )

    raise UnknownProviderError(f"Unknown LLM provider: '{cfg.LLM_PROVIDER}'")


@lru_cache
def get_llm_provider() -> BaseLLMProvider:
    """FastAPI dependency — singleton across the app's lifetime."""
    return create_llm_provider(settings)
