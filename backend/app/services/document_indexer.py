from pathlib import Path
from uuid import UUID, uuid4

from ..core import (
    ChunkingStrategy,
    DistanceMetric,
    DocumentStatus,
    IndexingError,
)
from ..embedders import BaseEmbeddingProvider
from ..schemas import Chunk, IndexingResponse
from ..vectorstores import BaseVectorStore, PointData, PointPayload
from .document_chunker import chunk_document
from .document_embedder import embed_chunks


def _build_points(
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[PointData]:
    """Zip chunks + vectors into provider-agnostic points."""
    if len(chunks) != len(vectors):
        raise IndexingError(
            f"Vector/chunk count mismatch: {len(vectors)} vectors for "
            f"{len(chunks)} chunks."
        )

    return [
        PointData(
            id=uuid4(),
            vector=vector,
            payload=PointPayload(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_type=chunk.source_type,
                content_hash=chunk.content_hash,
                parent_chunk_id=chunk.parent_chunk_id,
                parent_content=chunk.parent_content,
            ),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


async def index_document(
    document_id: UUID,
    upload_dir: Path,
    provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    collection_name: str,
    distance: DistanceMetric,
    strategy: ChunkingStrategy,
    embedding_chunk_size: int,
    embedding_overlap: int,
    prompt_chunk_size: int | None = None,
    prompt_overlap: int | None = None,
) -> IndexingResponse:
    """Parse -> Chunk -> Embed -> Upsert pipeline.

    Pure orchestration: this function never touches Qdrant, HTTP, or
    FastAPI. Vendor concerns stay inside `vector_store`; transport
    concerns stay inside the API layer.
    """
    if embedding_chunk_size > provider.max_sequence_length:
        raise IndexingError(
            f"Requested embedding_chunk_size ({embedding_chunk_size}) exceeds model "
            f"maximum context length ({provider.max_sequence_length})."
        )

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        prompt_chunk_size=prompt_chunk_size,
        prompt_overlap=prompt_overlap,
    )

    if not chunks:
        raise IndexingError(f"Document '{document_id}' produced no chunks to index.")

    # --- Cross-document exact duplicate prevention (content hash) ---
    candidate_hashes = frozenset(c.content_hash for c in chunks)
    existing_hashes = await vector_store.get_existing_hashes(
        collection_name, candidate_hashes
    )
    new_chunks = [c for c in chunks if c.content_hash not in existing_hashes]

    if new_chunks:
        vectors = await embed_chunks(new_chunks, provider)
        points = _build_points(new_chunks, vectors)
    else:
        vectors = []
        points = []
    dimension = provider.embedding_dimension

    await vector_store.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        distance=distance,
    )

    await vector_store.delete_by_document(
        collection_name=collection_name,
        document_id=str(document_id),
    )

    indexed = await vector_store.upsert(
        collection_name=collection_name,
        points=points,
    )
    return IndexingResponse(
        document_id=document_id,
        collection_name=collection_name,
        total_chunks=len(chunks),
        indexed_points=indexed,
        embedding_model=provider.model_name,
        dimension=dimension,
        status=DocumentStatus.INDEXED,
    )
