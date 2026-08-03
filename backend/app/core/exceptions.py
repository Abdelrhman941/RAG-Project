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
"""
from __future__ import annotations

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


class EmptyFileError(DocumentError):
    """Raised when an uploaded file has zero bytes."""


class FileTooLargeError(DocumentError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"File exceeds maximum size of {max_bytes} bytes.")


class DocumentNotFoundError(DocumentError):
    """Raised when a document cannot be found."""

    def __init__(self, document_id: UUID):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' was not found.")


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when no parser supports the given document type."""

    def __init__(self, extension: str):
        self.extension = extension
        super().__init__(f"No parser registered for extension '{extension}'.")


class ParsingError(DocumentError):
    """Raised when document parsing fails."""


# -------------- Chunking ----------------------------
class UnsupportedChunkingStrategyError(DocumentError):
    """Raised when no chunker supports the requested strategy."""

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        super().__init__(f"No chunker registered for strategy '{strategy.value}'.")


class InvalidChunkingParametersError(DocumentError):
    """Raised when chunking parameters are invalid."""


# -------------- Embedding ----------------------------
class UnsupportedEmbeddingProviderError(DocumentError):
    """Raised when no embedding provider matches the configured provider."""

    def __init__(self, provider: EmbeddingProviderName):
        self.provider = provider
        super().__init__(f"No embedding provider registered for '{provider.value}'.")


class EmbeddingError(DocumentError):
    """Raised when embedding generation fails."""


# -------------- Vector Store / Indexing ----------------------------
class VectorStoreError(DocumentError):
    """Base exception for vector store failures."""


class UnsupportedVectorStoreProviderError(VectorStoreError):
    """Raised when no vector store provider matches the configured provider."""

    def __init__(self, provider: VectorStoreProvider):
        self.provider = provider
        super().__init__(f"No vector store registered for '{provider.value}'.")


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when the configured vector store is unreachable."""


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when embedding dimension differs from collection dimension."""

    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Embedding dimension {actual} does not match collection "
            f"dimension {expected}."
        )


class IndexingError(VectorStoreError):
    """Raised when indexing vectors fails."""


# -------------- Retrieval ----------------------------
class RetrievalError(VectorStoreError):
    """Raised when semantic retrieval fails."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when the requested collection does not exist."""

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
