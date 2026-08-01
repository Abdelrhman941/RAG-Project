import logging
from pathlib import Path
from uuid import UUID

from anyio import to_thread

from ..core import ChunkingStrategy, DocumentStatus
from ..embedders import BaseEmbeddingProvider
from ..schemas import EmbeddingResponse
from .document_chunker import chunk_document

logger = logging.getLogger(__name__)


async def embed_document(
    document_id: UUID,
    upload_dir: Path,
    provider: BaseEmbeddingProvider,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
) -> EmbeddingResponse:
    """Parse -> Chunk -> Embed pipeline. Stateless: nothing is persisted here."""
    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    texts = [chunk.content for chunk in chunks]

    try:
        await to_thread.run_sync(provider.embed, texts)
    except Exception:
        logger.exception("Embedding failed")
        raise

    return EmbeddingResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        embedding_model=provider.model_name,
        dimension=provider.dimension,
        status=DocumentStatus.COMPLETED,
    )
