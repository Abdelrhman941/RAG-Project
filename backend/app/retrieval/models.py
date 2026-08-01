from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """Domain-level search hit returned by a `BaseVectorStore.search()` call.

    Introduced (Retrieval). The service layer depends on this
    model, NEVER on Qdrant's `ScoredPoint`. Each vector store adapter is
    responsible for mapping its native hit type into this shape at the
    adapter boundary, keeping the service free of any vendor imports.
    """

    model_config = ConfigDict(frozen=True)

    document_id: UUID4
    chunk_id: UUID4
    chunk_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(ge=1)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    content: str
