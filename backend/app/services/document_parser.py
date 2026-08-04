from collections.abc import AsyncIterator
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
) -> tuple[AsyncIterator[str], SourceType]:
    """Parse a stored document into a stream of page texts and its SourceType."""

    path = find_document_path(
        document_id=document_id,
        upload_dir=upload_dir,
    )
    try:
        extension = DocumentExtension(path.suffix.lower())
    except ValueError:
        raise UnsupportedDocumentTypeError(path.suffix) from None

    parser = get_parser(extension)
    sync_gen = parser.parse(path)

    async def _async_gen() -> AsyncIterator[str]:
        sentinel = object()
        while True:
            item = await to_thread.run_sync(next, sync_gen, sentinel)
            if item is sentinel:
                break
            yield str(item)

    source_type = SourceType(extension.value.lstrip("."))

    return _async_gen(), source_type
