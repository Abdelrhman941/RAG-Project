from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..schemas import SparseVector


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Embedding vector dimension."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Maximum number of tokens the model can process."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed one or more documents."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Embed a search query."""
        raise NotImplementedError


class BaseSparseEmbeddingProvider(ABC):
    """Abstract base class for sparse embedding providers (e.g. SPLADE, BM25)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying sparse model."""
        raise NotImplementedError

    @abstractmethod
    def embed_sparse_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseVector]:
        """Embed one or more documents into sparse vectors."""
        raise NotImplementedError

    @abstractmethod
    def embed_sparse_query(
        self,
        text: str,
    ) -> SparseVector:
        """Embed a search query into a sparse vector."""
        raise NotImplementedError


class BaseRerankerProvider(ABC):
    """Abstract base class for cross-encoder rerankers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying reranker model."""
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        query: str,
        texts: Sequence[str],
    ) -> list[float]:
        """Score each (query, text) pair. Returns calibrated scores between 0 and 1."""
        raise NotImplementedError
