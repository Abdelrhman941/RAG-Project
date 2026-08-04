from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from ..core import DocumentStatus


class IndexingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    collection_name: str
    total_chunks: Annotated[int, Field(ge=0)]
    indexed_points: Annotated[int, Field(ge=0)]
    embedding_model: str
    dimension: Annotated[int, Field(gt=0)]
    status: DocumentStatus
