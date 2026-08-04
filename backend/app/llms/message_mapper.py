from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from ..generation import ChatMessage, ChatRole
from .base import ProviderChatMessage

PromptMessage: TypeAlias = ChatMessage | Mapping[str, str | ChatRole]


def build_provider_message(
    *,
    role: ChatRole | str,
    content: str,
) -> ProviderChatMessage:
    """Create a provider-compatible chat message."""
    if isinstance(role, ChatRole):
        role = role.value

    match role:
        case "system":
            return {"role": "system", "content": content}
        case "user":
            return {"role": "user", "content": content}
        case "assistant":
            return {"role": "assistant", "content": content}
        case _:
            raise ValueError(f"Unsupported chat role: {role!r}")


def to_provider_messages(
    messages: Sequence[PromptMessage],
) -> list[ProviderChatMessage]:
    """Convert domain messages into provider messages."""
    provider_messages: list[ProviderChatMessage] = []

    for message in messages:
        if isinstance(message, Mapping):
            role = message["role"]
            content = str(message["content"])
        else:
            role = message.role
            content = message.content

        provider_messages.append(
            build_provider_message(
                role=role,
                content=content,
            )
        )

    return provider_messages
