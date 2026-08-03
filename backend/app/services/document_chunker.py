import uuid
from pathlib import Path
from uuid import UUID

from ..chunkers import get_chunker
from ..core import ChunkingStrategy, InvalidChunkingParametersError
from ..schemas import Chunk
from .document_parser import parse_document


def _validate_parameters(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkingParametersError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise InvalidChunkingParametersError(
            "overlap must be greater than or equal to 0."
        )
    if overlap >= chunk_size:
        raise InvalidChunkingParametersError("overlap must be smaller than chunk_size.")


async def chunk_document(
    document_id: UUID,
    upload_dir: Path,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Find -> Parse -> Chunk pipeline.

    Re-uses parse_document rather than reading the raw file again: the
    Parser is the single source of truth for extracted text, and the
    Chunker only ever operates on that output.
    """
    _validate_parameters(chunk_size, overlap)

    pages = await parse_document(document_id, upload_dir)
    chunker = get_chunker(strategy)

    chunks: list[Chunk] = []
    chunk_index = 0
    for page_number, page_text in enumerate(pages, start=1):
        for content, start_char, end_char in chunker.chunk(
            page_text, chunk_size, overlap
        ):
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4(),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    content=content,
                    start_char=start_char,
                    end_char=end_char,
                    char_count=len(content),
                )
            )
            chunk_index += 1

    return chunks
