"""Public wire schemas for the generation layer (Sprint 9).

These are the shapes the `/chat` API exposes to clients. Kept separate
from the internal `app.generation.models` domain models — mirroring how
`RetrievedChunk` (schema) is separate from `SearchResult` (domain).

Phase 4 additions: `ChatRequest` / `ChatResponse` — the actual request
and response contract for `POST /api/v1/chat`. `GenerationResponse`,
`ChatMessageSchema` and `TokenUsageSchema` are left untouched from
Phase 1; they're placeholders for a future streaming/memory phase and
aren't returned by `/chat` today (Groq doesn't report token usage
through `BaseLLMProvider.generate()` yet, so a `usage` field would
always be zero — better to omit it than fake it).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field, field_validator

from ..core import get_settings

_settings = get_settings()


class ChatRoleSchema(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessageSchema(BaseModel):
    """Wire representation of a chat message."""

    model_config = ConfigDict(frozen=True)

    role: ChatRoleSchema
    content: Annotated[str, Field(min_length=1)]


class CitationSchema(BaseModel):
    """Wire representation of a citation for the final API response."""

    model_config = ConfigDict(frozen=True)

    document_id: UUID4
    chunk_id: UUID4
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class TokenUsageSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0


class GenerationResponse(BaseModel):
    """Generic generation response envelope (Phase 1 placeholder).

    Not used by `/chat` — see `ChatResponse` below, which matches the
    Phase 4 spec exactly (no `usage`, adds `total_citations`).
    """

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    model: str
    citations: list[CitationSchema] = Field(default_factory=list)
    usage: TokenUsageSchema = Field(default_factory=TokenUsageSchema)


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Client payload for `POST /api/v1/chat`.

    Mirrors `RetrievalRequest` on purpose: same validation shape, same
    defaults sourced from `Settings`, so the two endpoints feel
    consistent to API consumers.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    query: Annotated[str, Field(min_length=1)]
    top_k: int = Field(
        default=_settings.DEFAULT_TOP_K,
        ge=1,
        le=_settings.MAX_TOP_K,
        description="Maximum number of chunks to retrieve as context.",
    )

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return value


class ChatResponse(BaseModel):
    """Envelope returned by `POST /api/v1/chat`."""

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    model: str
    total_citations: Annotated[int, Field(ge=0)]
    citations: list[CitationSchema] = Field(default_factory=list)
