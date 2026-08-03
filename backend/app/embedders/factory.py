from functools import lru_cache

from ..core import (
    EmbeddingProviderName,
    Settings,
    UnsupportedEmbeddingProviderError,
    get_settings,
)
from .base import BaseEmbeddingProvider
from .sentence_transformer import SentenceTransformerProvider


def create_embedding_provider(
    settings: Settings,
) -> BaseEmbeddingProvider:
    """Create an embedding provider from application settings."""
    if settings.EMBEDDING_PROVIDER is EmbeddingProviderName.SENTENCE_TRANSFORMER:
        return SentenceTransformerProvider(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    raise UnsupportedEmbeddingProviderError(settings.EMBEDDING_PROVIDER)


@lru_cache
def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return the application-wide embedding provider singleton."""
    return create_embedding_provider(get_settings())
