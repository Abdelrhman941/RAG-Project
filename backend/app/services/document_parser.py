from pathlib import Path
from uuid import UUID

from anyio import to_thread

from ..core.enums.document import DocumentExtension
from ..core.exceptions import DocumentNotFoundError
from ..parsers import get_parser


def find_document_path(document_id: UUID, upload_dir: Path) -> Path:
    matches = list(upload_dir.glob(f"{document_id}.*"))
    if not matches:
        raise DocumentNotFoundError(document_id)
    return matches[0]


async def parse_document(document_id: UUID, upload_dir: Path) -> list[str]:
    path = find_document_path(document_id, upload_dir)
    extension = DocumentExtension(path.suffix.lower())
    parser = get_parser(extension)
    return await to_thread.run_sync(parser.parse, path)
