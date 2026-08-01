from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..core import DocumentStatus


class IndexingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    collection_name: str
    total_chunks: Annotated[int, Field(ge=0)]
    indexed_points: Annotated[int, Field(ge=0)]
    embedding_model: str
    dimension: Annotated[int, Field(gt=0)]
    status: DocumentStatus
