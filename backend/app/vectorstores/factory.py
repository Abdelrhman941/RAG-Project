from functools import lru_cache

from ..core import (
    Settings,
    UnsupportedVectorStoreProviderError,
    VectorStoreProvider,
    get_settings,
)
from .base import BaseVectorStore
from .qdrant import QdrantVectorStore


def create_vector_store(settings: Settings) -> BaseVectorStore:
    """Create a vector store from application settings."""
    if settings.VECTOR_STORE_PROVIDER is VectorStoreProvider.QDRANT:
        return QdrantVectorStore(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            api_key=settings.QDRANT_API_KEY,
        )

    raise UnsupportedVectorStoreProviderError(settings.VECTOR_STORE_PROVIDER)


@lru_cache
def get_vector_store() -> BaseVectorStore:
    """Return the application-wide vector store singleton."""
    return create_vector_store(get_settings())
