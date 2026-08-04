from abc import ABC, abstractmethod
from collections.abc import Iterator

from .models import ChunkingConfig, ChunkSpan


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
    ) -> Iterator[ChunkSpan]:
        """Split text into ordered chunk spans."""
        raise NotImplementedError
