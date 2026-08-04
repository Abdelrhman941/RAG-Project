from pathlib import Path
from uuid import UUID

from anyio import to_thread

from ..core import (
    DocumentExtension,
    DocumentNotFoundError,
    SourceType,
    UnsupportedDocumentTypeError,
)
from ..parsers import get_parser


def find_document_path(
    document_id: UUID,
    upload_dir: Path,
) -> Path:
    """Locate a stored document by its UUID."""

    match = next(upload_dir.glob(f"{document_id}.*"), None)
    if match is None:
        raise DocumentNotFoundError(document_id)
    return match


async def parse_document(
    document_id: UUID, upload_dir: Path
) -> tuple[list[str], SourceType]:
    """Parse a stored document into a list of page texts and its SourceType."""

    path = find_document_path(
        document_id=document_id,
        upload_dir=upload_dir,
    )
    try:
        extension = DocumentExtension(path.suffix.lower())
    except ValueError:
        raise UnsupportedDocumentTypeError(path.suffix) from None
        
    parser = get_parser(extension)
    pages = await to_thread.run_sync(parser.parse, path)
    source_type = SourceType(extension.value.lstrip("."))
    
    return pages, source_type
