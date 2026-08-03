from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker
from .models import ChunkingConfig, ChunkSpan


class RecursiveChunker(BaseChunker):
    """Token-aware recursive chunker.

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
        if not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
        )
        chunks = splitter.split_text(text)
        spans: list[ChunkSpan] = []
        search_start = 0

        for chunk in chunks:
            start = text.find(chunk, search_start)
            # fallback for whitespace normalization differences
            if start == -1:
                start = text.find(chunk)
            if start == -1:
                start = search_start
            end = start + len(chunk)
            spans.append(
                ChunkSpan(
                    content=chunk,
                    start_char=start,
                    end_char=end,
                )
            )
            search_start = end
        return spans
