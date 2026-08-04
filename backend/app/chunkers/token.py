import logging
from collections.abc import Iterator

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
    ) -> Iterator[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return

        import uuid

        if config.prompt_chunk_size is not None and config.prompt_overlap is not None:
            parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config.prompt_chunk_size,
                chunk_overlap=config.prompt_overlap,
            )
            child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config.embedding_chunk_size,
                chunk_overlap=config.embedding_overlap,
            )

            p_chunks = parent_splitter.split_text(text)
            p_tuples = self._compute_offsets(p_chunks, text)

            spans = []
            for p_content, p_start, _p_end in p_tuples:
                p_id = str(uuid.uuid4())
                c_chunks = child_splitter.split_text(p_content)
                c_tuples_rel = self._compute_offsets(c_chunks, p_content)
                merged_c_tuples = self._merge_small_chunks(
                    c_tuples_rel, config.min_chunk_chars
                )

                for c_content, c_rel_start, c_rel_end in merged_c_tuples:
                    spans.append(
                        ChunkSpan(
                            content=c_content,
                            start_char=p_start + c_rel_start,
                            end_char=p_start + c_rel_end,
                            source_type=config.source_type,
                            parent_chunk_id=p_id,
                            parent_content=p_content,
                        )
                    )
            yield from spans

        else:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config.embedding_chunk_size,
                chunk_overlap=config.embedding_overlap,
            )
            raw_chunks = splitter.split_text(text)
            raw_tuples = self._compute_offsets(raw_chunks, text)
            merged_tuples = self._merge_small_chunks(raw_tuples, config.min_chunk_chars)

            yield from [
                ChunkSpan(
                    content=t[0],
                    start_char=t[1],
                    end_char=t[2],
                    source_type=config.source_type,
                )
                for t in merged_tuples
            ]

    @staticmethod
    def _compute_offsets(
        chunks: list[str], source_text: str
    ) -> list[tuple[str, int, int]]:
        raw_tuples: list[tuple[str, int, int]] = []
        search_start = 0
        for chunk_text in chunks:
            start = source_text.find(chunk_text, search_start)
            if start == -1:
                start = source_text.find(chunk_text)
            if start == -1:
                logger.warning(
                    f"Could not find chunk in normalized text, dropping chunk. "
                    f"(starts with: {chunk_text[:30]!r})"
                )
                continue

            end = start + len(chunk_text)
            raw_tuples.append((chunk_text, start, end))
            search_start = end
        return raw_tuples

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
