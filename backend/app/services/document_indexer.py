from uuid import UUID

from anyio import to_thread

from ..core import (
    ChunkingStrategy,
    DistanceMetric,
    DocumentStatus,
    IndexingError,
)
from ..embedders import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from ..schemas import Chunk, IndexingResponse, SparseVector
from ..vectorstores import BaseVectorStore, PointData, PointPayload
from .document_chunker import DocumentChunkerService
from .document_embedder import DocumentEmbedderService


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


class DocumentIndexerService:
    def __init__(
        self,
        chunker: DocumentChunkerService,
        embedder: DocumentEmbedderService,
        provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        sparse_provider: BaseSparseEmbeddingProvider | None = None,
        ingestion_batch_size: int = 100,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._provider = provider
        self._vector_store = vector_store
        self._sparse_provider = sparse_provider
        self._ingestion_batch_size = ingestion_batch_size

    async def _process_batch(
        self,
        batch: list[Chunk],
        collection_name: str,
    ) -> int:
        """Process and upsert a batch of chunks."""
        candidate_hashes = frozenset(c.content_hash for c in batch)
        existing_hashes = await self._vector_store.get_existing_hashes(
            collection_name, candidate_hashes
        )
        new_chunks = [c for c in batch if c.content_hash not in existing_hashes]

        if not new_chunks:
            return 0

        vectors = await self._embedder.embed_chunks(new_chunks)
        sparse_vectors = None
        if self._sparse_provider:
            sparse_vectors = await to_thread.run_sync(
                self._sparse_provider.embed_sparse_documents,
                [c.content for c in new_chunks],
            )
        points = _build_points(new_chunks, vectors, sparse_vectors)

        indexed = await self._vector_store.upsert(
            collection_name=collection_name,
            points=points,
        )
        return indexed

    async def index_document(
        self,
        document_id: UUID,
        collection_name: str,
        distance: DistanceMetric,
        strategy: ChunkingStrategy,
        embedding_chunk_size: int,
        embedding_overlap: int,
        prompt_chunk_size: int | None = None,
        prompt_overlap: int | None = None,
    ) -> IndexingResponse:
        """Parse -> Chunk -> Embed -> Upsert pipeline."""
        if embedding_chunk_size > self._provider.max_sequence_length:
            raise IndexingError(
                f"Requested embedding_chunk_size ({embedding_chunk_size}) "
                f"exceeds model "
                f"maximum context length ({self._provider.max_sequence_length})."
            )

        chunks_stream = self._chunker.chunk_document(
            document_id=document_id,
            strategy=strategy,
            embedding_chunk_size=embedding_chunk_size,
            embedding_overlap=embedding_overlap,
            prompt_chunk_size=prompt_chunk_size,
            prompt_overlap=prompt_overlap,
        )

        dimension = self._provider.embedding_dimension
        await self._vector_store.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance=distance,
        )

        await self._vector_store.delete_by_document(
            collection_name=collection_name,
            document_id=str(document_id),
        )

        total_chunks = 0
        total_indexed = 0
        batch: list[Chunk] = []

        async for chunk in chunks_stream:
            batch.append(chunk)
            if len(batch) >= self._ingestion_batch_size:
                indexed = await self._process_batch(batch, collection_name)
                total_indexed += indexed
                total_chunks += len(batch)
                batch = []

        if batch:
            indexed = await self._process_batch(batch, collection_name)
            total_indexed += indexed
            total_chunks += len(batch)

        if total_chunks == 0:
            raise IndexingError(
                f"Document '{document_id}' produced no chunks to index."
            )

        return IndexingResponse(
            document_id=document_id,
            collection_name=collection_name,
            total_chunks=total_chunks,
            indexed_points=total_indexed,
            embedding_model=self._provider.model_name,
            dimension=dimension,
            status=DocumentStatus.INDEXED,
        )
