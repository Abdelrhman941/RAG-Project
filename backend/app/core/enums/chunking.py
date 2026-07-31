from enum import Enum


class ChunkingStrategy(str, Enum):
    # ----- Supported strategies -----
    CHARACTER = "character"
    # SENTENCE = "sentence"
    # TOKEN = "token"
    # SEMANTIC = "semantic"
