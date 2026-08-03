from enum import Enum


class DocumentExtension(str, Enum):
    """Supported document file extensions."""

    PDF = ".pdf"
    TXT = ".txt"
    MD = ".md"


class DocumentStatus(str, Enum):
    """Document processing status."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    COMPLETED = "completed"
    FAILED = "failed"
