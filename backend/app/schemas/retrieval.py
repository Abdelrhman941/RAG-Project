from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from .common import TopKQueryRequest


class RetrievalRequest(TopKQueryRequest):
    """Client payload for `POST /api/v1/retrieval/search`."""


class RetrievedChunk(BaseModel):
    """One ranked chunk in a retrieval response."""

    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    chunk_id: UUID4
    chunk_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    content: str


class RetrievalResponse(BaseModel):
    """Envelope returned by the retrieval endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str
    embedding_model: str
    total_results: Annotated[int, Field(ge=0)]
    results: list[RetrievedChunk]
