from enum import Enum


class ChunkingStrategy(str, Enum):
    """Chunking strategy used when splitting a document into chunks."""

    TOKEN = "token"

    ## Future strategies
    # SENTENCE = "sentence"
    # SEMANTIC = "semantic"
