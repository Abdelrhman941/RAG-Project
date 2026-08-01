from typing import Annotated

from fastapi import Depends

from ..core import Settings, get_settings
from ..embedders import BaseEmbeddingProvider, get_embedding_provider

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_current_embedding_provider(
    settings: SettingsDep,
) -> BaseEmbeddingProvider:
    return get_embedding_provider(settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)


EmbeddingProviderDep = Annotated[
    BaseEmbeddingProvider, Depends(get_current_embedding_provider)
]
