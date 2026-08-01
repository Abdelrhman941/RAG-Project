from enum import Enum


class DocumentExtension(str, Enum):
    # ----- Supported file types -----
    PDF = ".pdf"
    TXT = ".txt"
    MD = ".md"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    COMPLETED = "completed"
    FAILED = "failed"
