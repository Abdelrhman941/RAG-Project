import logging
import uuid
from pathlib import Path
from uuid import UUID

from anyio import to_thread

from ..core import (
    ChunkingStrategy,
    DistanceMetric,
    DocumentStatus,
    IndexingError,
)
from ..embedders import BaseEmbeddingProvider
from ..schemas import Chunk, IndexingResponse, PointData, PointPayload
from ..vectorstores import BaseVectorStore
from .document_chunker import chunk_document

logger = logging.getLogger(__name__)


def _build_points(chunks: list[Chunk], vectors: list[list[float]]) -> list[PointData]:
    """Zip chunks + vectors into domain-level points.

    Lives in the service layer because it is business logic (how do we
    represent a stored chunk?), not vendor logic. The vector store
    adapter is what maps `PointData` -> `PointStruct` at the boundary.
    """
    if len(chunks) != len(vectors):
        raise IndexingError(
            f"Vector/chunk count mismatch: {len(vectors)} vectors for "
            f"{len(chunks)} chunks."
        )

    return [
        PointData(
            id=uuid.uuid4(),
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
    # 1) Parse + Chunk (reuses the same source of truth as /chunks).
    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        # A document that parses to no chunks should not silently succeed.
        raise IndexingError(f"Document '{document_id}' produced no chunks to index.")

    # 2) Embed. The provider is a sync CPU-bound API, so we push it off
    # the event loop the same way `embed_document` does.
    texts = [chunk.content for chunk in chunks]
    try:
        vectors = await to_thread.run_sync(provider.embed, texts)
    except Exception:
        logger.exception("Embedding failed during indexing")
        raise

    # 3) Build provider-agnostic points.
    points = _build_points(chunks, vectors)

    # 4) Ensure the collection exists with the right dimension.
    await vector_store.create_collection(
        collection_name=collection_name,
        dimension=provider.dimension,
        distance=distance,
    )

    # 5) Idempotency: wipe previous points for this document before
    # writing the fresh set. Re-indexing the same doc should not
    # accumulate duplicates.
    await vector_store.delete_by_document(
        collection_name=collection_name,
        document_id=str(document_id),
    )

    # 6) Upsert.
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
        dimension=provider.dimension,
        status=DocumentStatus.INDEXED,
    )
