from __future__ import annotations

from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from .common import TopKQueryRequest


class CitationSchema(BaseModel):
    """Wire representation of a citation for the final API response."""

    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    chunk_id: UUID4
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------
class ChatRequest(TopKQueryRequest):
    """Client payload for `POST /api/v1/chat`.

    Mirrors `RetrievalRequest` on purpose: same validation shape, same
    defaults sourced from `Settings`, so the two endpoints feel
    consistent to API consumers.
    """


class ChatResponse(BaseModel):
    """Envelope returned by `POST /api/v1/chat`."""

    model_config = ConfigDict(frozen=True)
    query: str
    answer: str
    model: str
    total_citations: Annotated[int, Field(ge=0)]
    citations: list[CitationSchema] = Field(default_factory=list)
