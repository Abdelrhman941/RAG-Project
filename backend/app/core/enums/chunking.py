from enum import Enum


class ChunkingStrategy(str, Enum):
    """Chunking strategy used when splitting a document into chunks."""

    CHARACTER = "character"
    SENTENCE = "sentence"
    TOKEN = "token"
    SEMANTIC = "semantic"
