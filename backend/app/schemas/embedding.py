from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from ..core import DocumentStatus


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    total_chunks: Annotated[int, Field(ge=0)]
    embedding_model: str
    dimension: Annotated[int, Field(gt=0)]
    status: DocumentStatus
