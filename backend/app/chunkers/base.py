from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Interface every chunking strategy must implement.

    A chunker operates on a single page's text at a time; the caller
    (ChunkingService) is responsible for iterating over pages and
    stitching the resulting spans into full Chunk objects. This keeps
    page_number bookkeeping out of the chunking algorithm itself.
    """

    @abstractmethod
    def chunk(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        """Split `text` into (content, start_char, end_char) spans."""
