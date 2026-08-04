import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker
from .models import ChunkingConfig, ChunkSpan
from .normalizer import normalize_text

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Token-aware recursive chunker with text normalization and small-chunk merging.

    Uses LangChain's RecursiveCharacterTextSplitter configured with a
    tokenizer-aware length function. The splitter tries to preserve
    paragraphs, lines and sentences before falling back to characters,
    producing higher-quality chunks for RAG than a plain token splitter.
    """

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.embedding_chunk_size,
            chunk_overlap=config.embedding_overlap,
        )
        raw_chunks = splitter.split_text(text)

        # 1. Compute raw offsets before merging
        raw_tuples: list[tuple[str, int, int]] = []
        search_start = 0
        for chunk_text in raw_chunks:
            start = text.find(chunk_text, search_start)
            if start == -1:
                start = text.find(chunk_text)
            if start == -1:
                logger.warning(
                    f"Could not find chunk in normalized text, dropping chunk. "
                    f"(starts with: {chunk_text[:30]!r})"
                )
                continue

            end = start + len(chunk_text)
            raw_tuples.append((chunk_text, start, end))
            search_start = end

        # 2. Merge small chunks
        merged_tuples = self._merge_small_chunks(raw_tuples, config.min_chunk_chars)

        # 3. Build spans
        return [
            ChunkSpan(
                content=t[0],
                start_char=t[1],
                end_char=t[2],
                source_type=config.source_type,
            )
            for t in merged_tuples
        ]

    @staticmethod
    def _merge_small_chunks(
        chunks: list[tuple[str, int, int]], min_chars: int
    ) -> list[tuple[str, int, int]]:
        """Merge chunks whose character length is below *min_chars*.
        Operates on (content, start, end) tuples.
        """
        if not chunks:
            return chunks

        result: list[tuple[str, int, int]] = []

        # Pass 1: Left merge
        for chunk in chunks:
            content, start, end = chunk
            if len(content.strip()) >= min_chars:
                result.append(chunk)
            elif result:
                # Merge into previous
                prev_content, prev_start, _ = result[-1]
                result[-1] = (prev_content + "\n" + content, prev_start, end)
            else:
                # No previous chunk, defer for pass 2
                result.append(chunk)

        # Pass 2: Right merge any remaining leading small chunk
        if len(result) > 1 and len(result[0][0].strip()) < min_chars:
            c1, s1, _ = result[0]
            c2, _, e2 = result[1]
            result[1] = (c1 + "\n" + c2, s1, e2)
            result = result[1:]

        # Pass 3: Drop micro-chunks unless it's the sole remaining chunk
        if len(result) == 1:
            return result

        return [c for c in result if len(c[0].strip()) >= min_chars]
