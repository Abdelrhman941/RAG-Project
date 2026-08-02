"""Generation domain models (Sprint 9, Phase 1).

These are INTERNAL domain models. They know nothing about:
    - HTTP / FastAPI
    - Groq / OpenAI / any LLM SDK
    - Streaming / memory / multi-turn

The wire contract (`schemas.generation`) is a separate layer that
maps these into the public API response — mirroring how
`SearchResult` (domain) maps into `RetrievedChunk` (schema).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field


class ChatRole(str, Enum):
    """Roles a chat message can carry."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Provider-agnostic chat message."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: Annotated[str, Field(min_length=1)]


class Citation(BaseModel):
    """Evidence metadata for one retrieved chunk.

    Mirrors the retrieval payload. Do not invent fields (such as
    `source`) unless retrieval actually provides them.
    """

    model_config = ConfigDict(frozen=True)

    document_id: UUID4
    chunk_id: UUID4
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class TokenUsage(BaseModel):
    """LLM token accounting."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0


class GenerationResult(BaseModel):
    """Internal result produced by the generation pipeline."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
