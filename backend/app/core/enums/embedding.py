from enum import Enum


class EmbeddingProviderName(str, Enum):
    # ----- Supported providers -----
    SENTENCE_TRANSFORMER = "sentence_transformer"
    # OPENAI = "openai"
