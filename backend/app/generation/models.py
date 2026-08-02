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
    """Roles a chat message can carry.

    Kept intentionally minimal for Phase 1 (system + user). `ASSISTANT`
    is included because `GenerationResult` conceptually represents an
    assistant reply — even though Phase 1 does not yet call an LLM.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Provider-agnostic on purpose: any LLM SDK adapter (Groq, OpenAI,
    Anthropic, ...) is responsible for mapping this into whatever
    shape its client expects, at the adapter boundary.
    """

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: Annotated[str, Field(min_length=1)]


class Citation(BaseModel):
    """Provenance for a single piece of retrieved evidence used to answer.

    Domain model — NOT the API response shape. The API layer will
    project this into a public schema when `/chat` ships in a later
    phase.
    """

    model_config = ConfigDict(frozen=True)

    document_id: UUID4
    chunk_id: UUID4
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class TokenUsage(BaseModel):
    """LLM token accounting for one generation call.

    Not populated in Phase 1 (no LLM yet) but modelled now so downstream
    phases can slot it in without touching `GenerationResult`'s shape.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0


class GenerationResult(BaseModel):
    """Internal outcome of a generation call.

    Phase 1 does not produce this yet — it exists so the PromptBuilder
    and later Generator service share a single, stable contract.
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
