from dataclasses import dataclass

from ..core import ChunkingStrategy, InvalidChunkingParametersError, SourceType


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """A chunk and its character offsets within the original text."""

    content: str
    start_char: int
    end_char: int
    source_type: SourceType


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration shared by all chunking strategies."""

    strategy: ChunkingStrategy
    chunk_size: int
    overlap: int
    min_chunk_chars: int
    source_type: SourceType

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise InvalidChunkingParametersError("chunk_size must be greater than 0.")

        if self.overlap < 0:
            raise InvalidChunkingParametersError(
                "overlap must be greater than or equal to 0."
            )

        if self.overlap >= self.chunk_size:
            raise InvalidChunkingParametersError(
                "overlap must be smaller than chunk_size."
            )

        if self.min_chunk_chars <= 0:
            raise InvalidChunkingParametersError(
                "min_chunk_chars must be greater than 0."
            )
