from enum import Enum


class DocumentExtension(str, Enum):
    # ----- Supported file types -----
    PDF = ".pdf"
    TXT = ".txt"
    MD = ".md"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    COMPLETED = "completed"
    FAILED = "failed"
    # PARSING = "parsing"
    # CHUNKING = "chunking"
    # EMBEDDING = "embedding"
    # INDEXING = "indexing"
