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
    embedding_chunk_size: int,
    embedding_overlap: int,
    prompt_chunk_size: int | None = None,
    prompt_overlap: int | None = None,
) -> EmbeddingResponse:
    """Parse -> Chunk -> Embed pipeline.

    Stateless: nothing is persisted here.
    """
    if embedding_chunk_size > provider.max_sequence_length:
        raise EmbeddingError(
            f"Requested embedding_chunk_size ({embedding_chunk_size}) exceeds model "
            f"maximum context length ({provider.max_sequence_length})."
        )

    chunks_gen = chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        prompt_chunk_size=prompt_chunk_size,
        prompt_overlap=prompt_overlap,
    )
    chunks = [c async for c in chunks_gen]

    await embed_chunks(chunks, provider)
    return EmbeddingResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        embedding_model=provider.model_name,
        dimension=provider.embedding_dimension,
        status=DocumentStatus.EMBEDDING,
    )
