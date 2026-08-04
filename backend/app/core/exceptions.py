"""
Project exception hierarchy.

Exception
└── DocumentError (Main Root Exception for the project) "Here"
    │
    ├── EmptyFileError
    ├── FileTooLargeError
    ├── DocumentNotFoundError
    ├── UnsupportedDocumentTypeError
    ├── ParsingError
    │
    ├── UnsupportedChunkingStrategyError
    ├── InvalidChunkingParametersError
    │
    ├── UnsupportedEmbeddingProviderError
    ├── EmbeddingError
    │
    ├── VectorStoreError (Base Exception for Vector Stores) "Here"
    │   ├── UnsupportedVectorStoreProviderError
    │   ├── VectorStoreUnavailableError
    │   ├── VectorDimensionMismatchError
    │   ├── IndexingError
    │   ├── RetrievalError
    │   └── CollectionNotFoundError
    │
    └── LLMError (Base Exception for LLMs)   "Here"
        ├── LLMConfigError                   "Here"
        │   └── UnknownProviderError
        └── LLMProviderError

Every exception carries `status_code`, the HTTP status the global
handler in `main.py` returns for it. Subclasses that don't set their
own inherit `DocumentError.status_code` (500) — see `LLMConfigError`
and `UnknownProviderError`, which previously had no registered handler
at all and silently fell through to FastAPI's default 500.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from .enums import (
    ChunkingStrategy,
    EmbeddingProviderName,
    LLMProviderName,
    VectorStoreProvider,
)


# -------------- Document ----------------------------
class DocumentError(Exception):
    """Base exception for document ingestion pipeline."""

    status_code: int = 500


class EmptyFileError(DocumentError):
    """Raised when an uploaded file has zero bytes."""

    status_code = 400


class FileTooLargeError(DocumentError):
    """Raised when an uploaded file exceeds the configured size limit."""

    status_code = 413

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"File exceeds maximum size of {max_bytes} bytes.")


class DocumentNotFoundError(DocumentError):
    """Raised when a document cannot be found."""

    status_code = 404

    def __init__(self, document_id: UUID):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' was not found.")


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when no parser supports the given document type."""

    status_code = 415

    def __init__(self, extension: str, allowed: Iterable[str] | None = None):
        self.extension = extension
        message = f"Unsupported file type '{extension}'."
        if allowed:
            message += f" Allowed: {', '.join(allowed)}."
        super().__init__(message)


class ParsingError(DocumentError):
    """Raised when document parsing fails."""

    status_code = 500


# -------------- Chunking ----------------------------
class UnsupportedChunkingStrategyError(DocumentError):
    """Raised when no chunker supports the requested strategy."""

    status_code = 501

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        super().__init__(f"No chunker registered for strategy '{strategy.value}'.")


class InvalidChunkingParametersError(DocumentError):
    """Raised when chunking parameters are invalid."""

    status_code = 422


# -------------- Embedding ----------------------------
class UnsupportedEmbeddingProviderError(DocumentError):
    """Raised when no embedding provider matches the configured provider."""

    status_code = 500

    def __init__(self, provider: EmbeddingProviderName):
        self.provider = provider
        super().__init__(f"No embedding provider registered for '{provider.value}'.")


class EmbeddingError(DocumentError):
    """Raised when embedding generation fails."""

    status_code = 500


# -------------- Vector Store / Indexing ----------------------------
class VectorStoreError(DocumentError):
    """Base exception for vector store failures."""


class UnsupportedVectorStoreProviderError(VectorStoreError):
    """Raised when no vector store provider matches the configured provider."""

    status_code = 500

    def __init__(self, provider: VectorStoreProvider):
        self.provider = provider
        super().__init__(f"No vector store registered for '{provider.value}'.")


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when the configured vector store is unreachable."""

    status_code = 503


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when embedding dimension differs from collection dimension."""

    status_code = 409

    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Embedding dimension {actual} does not match collection "
            f"dimension {expected}."
        )


class IndexingError(VectorStoreError):
    """Raised when indexing vectors fails."""

    status_code = 500


# -------------- Retrieval ----------------------------
class RetrievalError(VectorStoreError):
    """Raised when semantic retrieval fails."""

    status_code = 500


class CollectionNotFoundError(VectorStoreError):
    """Raised when the requested collection does not exist."""

    status_code = 404

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' does not exist.")


# -------------- LLM ----------------------------
class LLMError(DocumentError):
    """Base exception for all LLM-related failures."""


class LLMConfigError(LLMError):
    """Raised when LLM configuration is invalid."""


class UnknownProviderError(LLMConfigError):
    """Raised when the configured LLM provider is unsupported."""

    def __init__(self, provider: LLMProviderName | str):
        self.provider = provider

        provider_name = (
            provider.value if isinstance(provider, LLMProviderName) else provider
        )

        super().__init__(f"No LLM provider registered for '{provider_name}'.")


class LLMProviderError(LLMError):
    """Raised when an LLM provider request fails."""

    status_code = 500
