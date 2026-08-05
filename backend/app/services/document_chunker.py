import hashlib
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from ..chunkers import ChunkingConfig, get_chunker
from ..chunkers.dedup import is_duplicate
from ..core import ChunkingStrategy
from ..schemas import Chunk
from .document_parser import DocumentParserService


class DocumentChunkerService:
    def __init__(
        self,
        parser: DocumentParserService,
        min_chunk_chars: int = 100,
        dedup_similarity_threshold: float = 0.9,
    ) -> None:
        self._parser = parser
        self._min_chunk_chars = min_chunk_chars
        self._dedup_similarity_threshold = dedup_similarity_threshold

    async def chunk_document(
        self,
        document_id: UUID,
        strategy: ChunkingStrategy,
        embedding_chunk_size: int,
        embedding_overlap: int,
        prompt_chunk_size: int | None = None,
        prompt_overlap: int | None = None,
    ) -> AsyncIterator[Chunk]:
        """Parse a document then split it into ordered, deduplicated chunks.

        Deduplication is intra-document only (string similarity via difflib).
        Cross-corpus duplicate prevention is handled separately.
        """
        async with self._parser.parse_document(document_id=document_id) as (
            pages,
            source_type,
        ):
            config = ChunkingConfig(
                strategy=strategy,
                embedding_chunk_size=embedding_chunk_size,
                embedding_overlap=embedding_overlap,
                prompt_chunk_size=prompt_chunk_size,
                prompt_overlap=prompt_overlap,
                min_chunk_chars=self._min_chunk_chars,
                source_type=source_type,
            )
            chunker = get_chunker(config.strategy)
            history: list[Chunk] = []
            current_chunk_index = 0
            page_number = 1

            async for page_text in pages:
                if not page_text.strip():
                    page_number += 1
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
                        parent_chunk_id=UUID(span.parent_chunk_id)
                        if span.parent_chunk_id
                        else None,
                        parent_content=span.parent_content,
                    )

                    if is_duplicate(
                        candidate, history, self._dedup_similarity_threshold
                    ):
                        continue

                    history.append(candidate)
                    if len(history) > 20:
                        history.pop(0)

                    yield candidate
                    current_chunk_index += 1

            page_number += 1
