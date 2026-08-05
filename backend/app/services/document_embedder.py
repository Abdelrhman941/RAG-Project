from collections.abc import Sequence

from anyio import to_thread

from ..core import EmbeddingError
from ..embedders import BaseEmbeddingProvider
from ..schemas import Chunk


class DocumentEmbedderService:
    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self._provider = provider

    async def embed_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> list[list[float]]:
        """Embed chunk texts into vectors using the configured provider."""
        texts = [chunk.content for chunk in chunks]
        if not texts:
            return []

        vectors = await to_thread.run_sync(self._provider.embed_documents, texts)

        if len(vectors) != len(chunks):
            raise EmbeddingError(
                f"Embedding provider returned {len(vectors)} vectors for "
                f"{len(chunks)} chunks."
            )

        return vectors
