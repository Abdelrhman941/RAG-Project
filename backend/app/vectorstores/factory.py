from functools import lru_cache

from ..core import UnsupportedVectorStoreProviderError, VectorStoreProvider
from .base import BaseVectorStore
from .qdrant import QdrantVectorStore


@lru_cache
def get_vector_store(
    provider: VectorStoreProvider,
    host: str,
    port: int,
    grpc_port: int,
    prefer_grpc: bool,
    api_key: str | None,
) -> BaseVectorStore:
    """Build (and cache) a vector store for the given configuration.

    `lru_cache` keeps a single client instance per unique configuration
    so we don't open a new gRPC/HTTP connection on every request.
    """
    if provider == VectorStoreProvider.QDRANT:
        return QdrantVectorStore(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
        )
    raise UnsupportedVectorStoreProviderError(provider.value)
