from uuid import UUID

from fastapi import APIRouter, status

from ...core import DocumentStatus
from ...schemas import ParsedDocument
from ...services import parse_document
from ..deps import SettingsDep

parse_router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@parse_router.post(
    "/{document_id}/parse",
    status_code=status.HTTP_200_OK,
    response_model=ParsedDocument,
)
async def parse_uploaded_document(
    document_id: UUID,
    settings: SettingsDep,
) -> ParsedDocument:
    pages = await parse_document(document_id, settings.UPLOAD_DIR)
    return ParsedDocument(
        document_id=document_id,
        status=DocumentStatus.PARSING,
        pages=pages,
    )
