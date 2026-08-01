from typing import Annotated

from fastapi import Depends

from ..core import Settings, get_settings
from ..embedders import BaseEmbeddingProvider, get_embedding_provider
from ..vectorstores import BaseVectorStore, get_vector_store

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_current_embedding_provider(
    settings: SettingsDep,
) -> BaseEmbeddingProvider:
    return get_embedding_provider(settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)


EmbeddingProviderDep = Annotated[
    BaseEmbeddingProvider, Depends(get_current_embedding_provider)
]


def get_current_vector_store(settings: SettingsDep) -> BaseVectorStore:
    return get_vector_store(
        provider=settings.VECTOR_STORE_PROVIDER,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=settings.QDRANT_PREFER_GRPC,
        api_key=settings.QDRANT_API_KEY,
    )


VectorStoreDep = Annotated[BaseVectorStore, Depends(get_current_vector_store)]
