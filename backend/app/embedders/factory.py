from functools import lru_cache

from ..core import (
    EmbeddingProviderName,
    Settings,
    UnsupportedEmbeddingProviderError,
    get_settings,
)
from .base import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    BaseSparseEmbeddingProvider,
)
from .cross_encoder import CrossEncoderReranker
from .fastembed_sparse import FastEmbedSparseProvider
from .sentence_transformer import SentenceTransformerProvider


def create_embedding_provider(
    settings: Settings,
) -> BaseEmbeddingProvider:
    """Create an embedding provider from application settings."""
    if settings.EMBEDDING_PROVIDER is EmbeddingProviderName.SENTENCE_TRANSFORMER:
        return SentenceTransformerProvider(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    raise UnsupportedEmbeddingProviderError(settings.EMBEDDING_PROVIDER)


@lru_cache
def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return the application-wide embedding provider singleton."""
    return create_embedding_provider(get_settings())


def create_reranker(
    settings: Settings,
) -> BaseRerankerProvider | None:
    """Create a reranker from application settings if configured."""
    if settings.RERANKER_MODEL_NAME:
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL_NAME)
    return None


@lru_cache
def get_reranker() -> BaseRerankerProvider | None:
    """Return the application-wide reranker singleton."""
    return create_reranker(get_settings())

def create_sparse_provider(
    settings: Settings,
) -> BaseSparseEmbeddingProvider | None:
    """Create a sparse provider from application settings if configured."""
    if settings.SPARSE_EMBEDDING_MODEL_NAME:
        return FastEmbedSparseProvider(model_name=settings.SPARSE_EMBEDDING_MODEL_NAME)
    return None

@lru_cache
def get_sparse_provider() -> BaseSparseEmbeddingProvider | None:
    """Return the application-wide sparse provider singleton."""
    return create_sparse_provider(get_settings())
