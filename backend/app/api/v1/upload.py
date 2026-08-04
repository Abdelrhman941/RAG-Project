import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from ...core import DocumentExtension, DocumentStatus, UnsupportedDocumentTypeError
from ...schemas import DocumentUploadResponse
from ...services import save_uploaded_file
from ..deps import SettingsDep

upload_router = APIRouter(prefix="/documents", tags=["Documents"])


@upload_router.post(
    "", status_code=status.HTTP_201_CREATED, response_model=DocumentUploadResponse
)
async def upload_document(
    settings: SettingsDep,
    file: Annotated[
        UploadFile,
        File(description="Upload a .pdf, .txt or .md document"),
    ],
) -> DocumentUploadResponse:
    # --- Validation 1: Check if a file was actually selected and sent ---
    if not file.filename:
        raise UnsupportedDocumentTypeError(
            "", allowed=[item.value for item in DocumentExtension]
        )

    # --- Validation 2: Extract the extension and verify it against allowed types ---
    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()
    try:
        extension_enum = DocumentExtension(extension)
    except ValueError:
        raise UnsupportedDocumentTypeError(
            extension, allowed=[item.value for item in DocumentExtension]
        ) from None

    file_id = uuid.uuid4()
    filename = f"{file_id}{extension_enum.value}"
    destination = settings.UPLOAD_DIR / filename

    # --- Validation 3: Save the file locally while validating the max file size ---
    size_bytes = await save_uploaded_file(
        file=file,
        destination=destination,
        max_size_bytes=settings.MAX_FILE_SIZE_BYTES,
    )

    return DocumentUploadResponse(
        id=file_id,
        filename=filename,
        original_filename=original_filename,
        extension=extension_enum,
        size_bytes=size_bytes,
        status=DocumentStatus.UPLOADED,
        uploaded_at=datetime.now(timezone.utc),
    )
