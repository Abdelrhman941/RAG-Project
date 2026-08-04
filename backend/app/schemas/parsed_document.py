from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from ..core import DocumentStatus


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    status: DocumentStatus
    pages: Annotated[list[str], Field(min_length=1)]
