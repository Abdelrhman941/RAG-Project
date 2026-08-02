"""
Unit tests for GroqProvider + factory.

Run: pytest -q app/tests/test_groq_provider.py
Requires: pytest-asyncio (add `asyncio_mode = "auto"` in pytest.ini/pyproject,
or use the @pytest.mark.asyncio decorators below if you prefer explicit mode).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError, UnknownProviderError
from app.llms.base import ChatMessage
from app.llms.factory import create_llm_provider
from app.llms.groq import GroqProvider


def _fake_groq_response(content: str | None = "Hello from Groq") -> SimpleNamespace:
    """Mimic the shape of Groq SDK's ChatCompletion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def provider() -> GroqProvider:
    return GroqProvider(api_key="test-key", model="llama-3.3-70b-versatile")


@pytest.mark.asyncio
async def test_generate_returns_string_answer(provider: GroqProvider) -> None:
    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_fake_groq_response("42")),
    ):
        result = await provider.generate([ChatMessage(role="user", content="hi")])

    assert isinstance(result, str)
    assert result == "42"


@pytest.mark.asyncio
async def test_generate_passes_messages_in_correct_shape(
    provider: GroqProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_groq_response())
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="hi"),
    ]

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        await provider.generate(messages)

    _, kwargs = mock_create.call_args
    assert kwargs["messages"] == messages
    assert kwargs["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_generate_raises_on_empty_content(provider: GroqProvider) -> None:
    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_fake_groq_response(content=None)),
    ):
        with pytest.raises(LLMProviderError):
            await provider.generate([ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_generate_wraps_sdk_errors(provider: GroqProvider) -> None:
    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=RuntimeError("network down")),
    ):
        with pytest.raises(LLMProviderError):
            await provider.generate([ChatMessage(role="user", content="hi")])


def test_factory_returns_groq_provider() -> None:
    cfg = Settings(LLM_PROVIDER="groq", GROQ_API_KEY="k", GROQ_MODEL="m")
    result = create_llm_provider(cfg)
    assert isinstance(result, GroqProvider)
    assert result.model_name == "m"


def test_factory_raises_on_unknown_provider() -> None:
    cfg = Settings(LLM_PROVIDER="does-not-exist", GROQ_API_KEY="k")
    with pytest.raises(UnknownProviderError):
        create_llm_provider(cfg)
