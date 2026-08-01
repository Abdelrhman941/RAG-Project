from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field, field_validator

from ..core import get_settings

_settings = get_settings()


class RetrievalRequest(BaseModel):
    """Client payload for `POST /api/v1/retrieval/search`.

    `query` is required and stripped of surrounding whitespace before
    validation. `top_k` is optional and falls back to the server default
    (`Settings.DEFAULT_TOP_K`).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    query: Annotated[
        str,
        Field(min_length=1),
    ]
    top_k: int = Field(
        default=_settings.DEFAULT_TOP_K,
        ge=1,
        le=_settings.MAX_TOP_K,
        description="Maximum number of chunks to return.",
    )

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        # `str_strip_whitespace=True` handles trimming; this guards against
        # queries that are whitespace-only (which would collapse to "").
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return value


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
