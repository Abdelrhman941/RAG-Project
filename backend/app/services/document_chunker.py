import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from ..chunkers import ChunkingConfig, get_chunker
from ..chunkers.dedup import is_duplicate
from ..core import ChunkingStrategy
from ..schemas import Chunk
from .document_parser import parse_document


async def chunk_document(
    document_id: UUID,
    upload_dir: Path,
    strategy: ChunkingStrategy,
    embedding_chunk_size: int,
    embedding_overlap: int,
) -> list[Chunk]:
    """Parse a document then split it into ordered, deduplicated chunks.

    Deduplication is intra-document only (string similarity via difflib).
    Cross-corpus duplicate prevention is handled separately.
    """

    from ..core import get_settings

    settings = get_settings()

    pages, source_type = await parse_document(
        document_id=document_id,
        upload_dir=upload_dir,
    )

    config = ChunkingConfig(
        strategy=strategy,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        min_chunk_chars=settings.MIN_CHUNK_CHARS,
        source_type=source_type,
    )
    chunker = get_chunker(config.strategy)
    chunks: list[Chunk] = []
    current_chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        for span in chunker.chunk(page_text, config):
            candidate = Chunk(
                chunk_id=uuid4(),
                document_id=document_id,
                chunk_index=current_chunk_index,
                page_number=page_number,
                content=span.content,
                start_char=span.start_char,
                end_char=span.end_char,
                char_count=len(span.content),
                source_type=span.source_type,
                content_hash=hashlib.sha256(span.content.encode()).hexdigest(),
            )

            if is_duplicate(candidate, chunks, settings.DEDUP_SIMILARITY_THRESHOLD):
                continue  # drop intra-document duplicate

            chunks.append(candidate)
            current_chunk_index += 1

    return chunks
