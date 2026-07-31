from datetime import datetime
from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from ..core import DocumentExtension, DocumentStatus


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID4
    status: DocumentStatus
    filename: str
    original_filename: str
    extension: DocumentExtension
    size_bytes: Annotated[int, Field(ge=0)]
    uploaded_at: datetime
