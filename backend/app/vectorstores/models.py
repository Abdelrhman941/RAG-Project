from typing import Annotated

from pydantic import UUID4, BaseModel, ConfigDict, Field

from ..core import SourceType


class PointPayload(BaseModel):
    """Metadata stored alongside the vector.

    Kept intentionally flat and provider-agnostic: the vector store
    adapter is responsible for serialising this into whatever payload
    shape the backing engine (Qdrant, pgvector, ...) expects.
    """

    model_config = ConfigDict(frozen=True)
    document_id: UUID4
    chunk_id: UUID4
    chunk_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(ge=1)]
    content: str
    source_type: SourceType
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=0)]
    content_hash: str  # SHA-256 hex digest of chunk content
    parent_chunk_id: UUID4 | None = None
    parent_content: str | None = None


class PointData(BaseModel):
    """Domain-level point: id + vector + payload.

    This is what the IndexingService hands to the VectorStore. The
    adapter converts it to the SDK-specific structure (e.g. Qdrant's
    `PointStruct`) at the very last step — keeping the service free of
    any vendor imports.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID4
    vector: Annotated[list[float], Field(min_length=1)]
    payload: PointPayload
