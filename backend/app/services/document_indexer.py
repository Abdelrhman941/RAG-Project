from pathlib import Path
from uuid import UUID

from ..core import (
    ChunkingStrategy,
    DistanceMetric,
    DocumentStatus,
    IndexingError,
)
from ..embedders import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from ..schemas import Chunk, IndexingResponse, SparseVector
from ..vectorstores import BaseVectorStore, PointData, PointPayload
from .document_chunker import chunk_document
from .document_embedder import embed_chunks


def _build_points(
    chunks: list[Chunk],
    vectors: list[list[float]],
    sparse_vectors: list[SparseVector] | None = None,
) -> list[PointData]:
    """Zip chunks + vectors into provider-agnostic points."""
    if len(chunks) != len(vectors):
        raise IndexingError(
            f"Vector/chunk count mismatch: {len(vectors)} vectors for "
            f"{len(chunks)} chunks."
        )

    return [
        PointData(
            id=chunk.chunk_id,
            vector=vector,
            sparse_vector=sparse_vectors[i] if sparse_vectors else None,
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
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]


async def _process_batch(
    batch: list[Chunk],
    vector_store: BaseVectorStore,
    collection_name: str,
    provider: BaseEmbeddingProvider,
    sparse_provider: BaseSparseEmbeddingProvider | None,
) -> int:
    """Process and upsert a batch of chunks."""
    candidate_hashes = frozenset(c.content_hash for c in batch)
    existing_hashes = await vector_store.get_existing_hashes(
        collection_name, candidate_hashes
    )
    new_chunks = [c for c in batch if c.content_hash not in existing_hashes]

    if not new_chunks:
        return 0

    vectors = await embed_chunks(new_chunks, provider)
    sparse_vectors = None
    if sparse_provider:
        from anyio import to_thread

        sparse_vectors = await to_thread.run_sync(
            sparse_provider.embed_sparse_documents, [c.content for c in new_chunks]
        )
    points = _build_points(new_chunks, vectors, sparse_vectors)

    indexed = await vector_store.upsert(
        collection_name=collection_name,
        points=points,
    )
    return indexed


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
    sparse_provider: BaseSparseEmbeddingProvider | None = None,
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

    chunks_stream = chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        prompt_chunk_size=prompt_chunk_size,
        prompt_overlap=prompt_overlap,
    )

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

    from ..core import get_settings

    settings = get_settings()

    total_chunks = 0
    total_indexed = 0
    batch: list[Chunk] = []

    async for chunk in chunks_stream:
        batch.append(chunk)
        if len(batch) >= settings.INGESTION_BATCH_SIZE:
            indexed = await _process_batch(
                batch, vector_store, collection_name, provider, sparse_provider
            )
            total_indexed += indexed
            total_chunks += len(batch)
            batch = []

    if batch:
        indexed = await _process_batch(
            batch, vector_store, collection_name, provider, sparse_provider
        )
        total_indexed += indexed
        total_chunks += len(batch)

    if total_chunks == 0:
        raise IndexingError(f"Document '{document_id}' produced no chunks to index.")

    return IndexingResponse(
        document_id=document_id,
        collection_name=collection_name,
        total_chunks=total_chunks,
        indexed_points=total_indexed,
        embedding_model=provider.model_name,
        dimension=dimension,
        status=DocumentStatus.INDEXED,
    )
