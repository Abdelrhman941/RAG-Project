from functools import lru_cache

from ..core import EmbeddingProviderName, UnsupportedEmbeddingProviderError
from .base import BaseEmbeddingProvider
from .sentence_transformer import SentenceTransformerProvider


@lru_cache
def get_embedding_provider(
    provider_name: EmbeddingProviderName, model_name: str
) -> BaseEmbeddingProvider:
    if provider_name == EmbeddingProviderName.SENTENCE_TRANSFORMER:
        return SentenceTransformerProvider(model_name)
    raise UnsupportedEmbeddingProviderError(provider_name.value)
