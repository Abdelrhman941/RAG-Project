from pathlib import Path
from uuid import UUID, uuid4

from ..chunkers import ChunkingConfig, get_chunker
from ..core import ChunkingStrategy
from ..schemas import Chunk
from .document_parser import parse_document


async def chunk_document(
    document_id: UUID,
    upload_dir: Path,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Parse a document then split it into ordered chunks."""

    from ..core import SourceType, get_settings

    settings = get_settings()

    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_chars=settings.MIN_CHUNK_CHARS,
        source_type=SourceType.TXT,  # Hardcoded temporarily until Task 6
    )
    pages = await parse_document(
        document_id=document_id,
        upload_dir=upload_dir,
    )
    chunker = get_chunker(config.strategy)
    chunks: list[Chunk] = []
    current_chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        for span in chunker.chunk(page_text, config):
            chunks.append(
                Chunk(
                    chunk_id=uuid4(),
                    document_id=document_id,
                    chunk_index=current_chunk_index,
                    page_number=page_number,
                    content=span.content,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    char_count=len(span.content),
                    source_type=span.source_type,
                )
            )
            current_chunk_index += 1
    return chunks
