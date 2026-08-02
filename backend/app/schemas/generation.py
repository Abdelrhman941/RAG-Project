"""Public wire schemas for the generation layer (Sprint 9, Phase 1).

These are the shapes the API will expose to clients when `/chat` ships
in a later phase. They are deliberately kept separate from the internal
`app.generation.models` domain models — mirroring how `RetrievedChunk`
(schema) is separate from `SearchResult` (domain).

Nothing in Phase 1 wires these into a router yet; they are declared
here so downstream phases can adopt them without touching the domain
layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field


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
    """Public generation response envelope.

    Placeholder for `/chat`. Not returned by any endpoint in Phase 1.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    model: str
    citations: list[CitationSchema] = Field(default_factory=list)
    usage: TokenUsageSchema = Field(default_factory=TokenUsageSchema)
