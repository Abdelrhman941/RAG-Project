from dataclasses import dataclass

from ..core import ChunkingStrategy, InvalidChunkingParametersError, SourceType


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """A chunk and its character offsets within the original text."""

    content: str
    start_char: int
    end_char: int
    source_type: SourceType
    parent_chunk_id: str | None = None
    parent_content: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration shared by all chunking strategies."""

    strategy: ChunkingStrategy
    embedding_chunk_size: int
    embedding_overlap: int
    min_chunk_chars: int
    source_type: SourceType
    prompt_chunk_size: int | None = None
    prompt_overlap: int | None = None
    max_prompt_chunk_size: int = 3000

    def __post_init__(self) -> None:
        if self.embedding_chunk_size <= 0:
            raise InvalidChunkingParametersError(
                "embedding_chunk_size must be greater than 0."
            )

        if self.embedding_overlap < 0:
            raise InvalidChunkingParametersError(
                "embedding_overlap must be greater than or equal to 0."
            )

        if self.embedding_overlap >= self.embedding_chunk_size:
            raise InvalidChunkingParametersError(
                "embedding_overlap must be smaller than embedding_chunk_size."
            )

        if self.prompt_chunk_size is not None:
            if self.prompt_chunk_size <= 0:
                raise InvalidChunkingParametersError(
                    "prompt_chunk_size must be greater than 0."
                )
            if self.prompt_chunk_size > self.max_prompt_chunk_size:
                raise InvalidChunkingParametersError(
                    f"prompt_chunk_size cannot exceed max_prompt_chunk_size "
                    f"({self.max_prompt_chunk_size})."
                )
            if (
                self.prompt_overlap is not None
                and self.prompt_overlap >= self.prompt_chunk_size
            ):
                raise InvalidChunkingParametersError(
                    "prompt_overlap must be smaller than prompt_chunk_size."
                )
            if self.embedding_chunk_size >= self.prompt_chunk_size:
                raise InvalidChunkingParametersError(
                    "embedding_chunk_size must be strictly smaller "
                    "than prompt_chunk_size."
                )

        if self.min_chunk_chars <= 0:
            raise InvalidChunkingParametersError(
                "min_chunk_chars must be greater than 0."
            )
