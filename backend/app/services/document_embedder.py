from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from anyio import to_thread

from ..core import ChunkingStrategy, DocumentStatus, EmbeddingError
from ..embedders import BaseEmbeddingProvider
from ..schemas import Chunk, EmbeddingResponse
from .document_chunker import chunk_document


async def embed_chunks(
    chunks: Sequence[Chunk],
    provider: BaseEmbeddingProvider,
) -> list[list[float]]:
    """Embed chunk texts into vectors using the configured provider."""
    texts = [chunk.content for chunk in chunks]
    if not texts:
        return []

    vectors = await to_thread.run_sync(provider.embed_documents, texts)

    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"Embedding provider returned {len(vectors)} vectors for "
            f"{len(chunks)} chunks."
        )

    return vectors


async def embed_document(
    document_id: UUID,
    upload_dir: Path,
    provider: BaseEmbeddingProvider,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
) -> EmbeddingResponse:
    """Parse -> Chunk -> Embed pipeline.

    Stateless: nothing is persisted here.
    """
    if chunk_size > provider.max_sequence_length:
        raise EmbeddingError(
            f"Requested chunk_size ({chunk_size}) exceeds model maximum context "
            f"length ({provider.max_sequence_length})."
        )

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    await embed_chunks(chunks, provider)
    return EmbeddingResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        embedding_model=provider.model_name,
        dimension=provider.embedding_dimension,
        status=DocumentStatus.EMBEDDING,
    )
