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


class SourceType(str, Enum):
    """The origin file type of a parsed document."""

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
