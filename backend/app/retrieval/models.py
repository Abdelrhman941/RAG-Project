from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """Provider-agnostic search result returned by the retrieval layer."""

    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    chunk_id: UUID4
    chunk_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    content: str
    parent_chunk_id: UUID4 | None = None
    parent_content: str | None = None
    rerank_score: float | None = None
