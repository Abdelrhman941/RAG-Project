from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    pages: Annotated[list[str], Field(min_length=1)]
