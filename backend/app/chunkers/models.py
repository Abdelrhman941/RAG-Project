from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """A chunk and its character offsets within the original text."""

    content: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration shared by all chunking strategies."""

    chunk_size: int
    overlap: int
