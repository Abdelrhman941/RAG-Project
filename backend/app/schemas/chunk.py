from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from ..core import ChunkingStrategy, DocumentStatus, SourceType


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: UUID4
    document_id: UUID4
    chunk_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(ge=1)]
    content: str
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=0)]
    char_count: Annotated[int, Field(ge=0)]
    source_type: SourceType
    content_hash: str  # SHA-256 hex digest of chunk content

    @model_validator(mode="after")
    def validate_span_consistency(self) -> "Chunk":
        if self.end_char < self.start_char:
            raise ValueError("end_char cannot be smaller than start_char.")
        if self.char_count != self.end_char - self.start_char:
            raise ValueError("char_count must equal end_char - start_char.")
        return self


class ChunkRequest(BaseModel):
    """Optional chunking overrides. Omitted fields fall back to server defaults."""

    strategy: ChunkingStrategy = ChunkingStrategy.TOKEN
    embedding_chunk_size: int | None = Field(default=None, gt=0)
    embedding_overlap: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_overlap_against_chunk_size(self) -> "ChunkRequest":
        if (
            self.embedding_chunk_size is not None
            and self.embedding_overlap is not None
            and self.embedding_overlap >= self.embedding_chunk_size
        ):
            raise ValueError(
                "embedding_overlap must be smaller than "
                "embedding_chunk_size."
            )
        return self


class ChunkResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    status: DocumentStatus
    total_chunks: int
    chunks: list[Chunk]
