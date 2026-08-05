from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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


class DocumentParserService:
    def __init__(self, upload_dir: Path) -> None:
        self._upload_dir = upload_dir

    def _find_document_path(self, document_id: UUID) -> Path:
        """Locate a stored document by its UUID."""
        match = next(self._upload_dir.glob(f"{document_id}.*"), None)
        if match is None:
            raise DocumentNotFoundError(document_id)
        return match

    @asynccontextmanager
    async def parse_document(
        self, document_id: UUID
    ) -> AsyncIterator[tuple[AsyncIterator[str], SourceType]]:
        """Parse a stored document into a stream of page texts and its SourceType.
        Must be used as an async context manager to ensure file handles are closed."""

        path = self._find_document_path(document_id=document_id)
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

        try:
            yield _async_gen(), source_type
        finally:
            sync_gen.close()
