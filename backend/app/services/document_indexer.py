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
    chunk_size: int,
    overlap: int,
) -> IndexingResponse:
    """Parse -> Chunk -> Embed -> Upsert pipeline.

    Pure orchestration: this function never touches Qdrant, HTTP, or
    FastAPI. Vendor concerns stay inside `vector_store`; transport
    concerns stay inside the API layer.
    """
    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise IndexingError(f"Document '{document_id}' produced no chunks to index.")

    vectors = await embed_chunks(chunks, provider)
    points = _build_points(chunks, vectors)
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
