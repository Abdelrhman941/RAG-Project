import pytest

from app.core.enums.vector_store import VectorStoreProvider
from app.core.exceptions import UnsupportedVectorStoreProviderError
from app.vectorstores import get_vector_store
from app.vectorstores.qdrant import QdrantVectorStore


def test_factory_returns_qdrant_for_qdrant_provider() -> None:
    store = get_vector_store(
        provider=VectorStoreProvider.QDRANT,
        host="localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        api_key=None,
    )
    assert isinstance(store, QdrantVectorStore)


def test_factory_caches_instances() -> None:
    a = get_vector_store(
        provider=VectorStoreProvider.QDRANT,
        host="localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        api_key=None,
    )
    b = get_vector_store(
        provider=VectorStoreProvider.QDRANT,
        host="localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        api_key=None,
    )
    assert a is b


def test_factory_raises_for_unknown_provider() -> None:
    class _Bogus:
        value = "bogus"

    with pytest.raises(UnsupportedVectorStoreProviderError):
        get_vector_store(
            provider=_Bogus(),  # type: ignore[arg-type]
            host="localhost",
            port=6333,
            grpc_port=6334,
            prefer_grpc=False,
            api_key=None,
        )
