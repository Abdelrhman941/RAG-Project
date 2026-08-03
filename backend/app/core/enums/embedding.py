from enum import Enum


class EmbeddingProviderName(str, Enum):
    """Embedding provider names."""

    SENTENCE_TRANSFORMER = "sentence_transformer"
