from abc import ABC, abstractmethod
from collections.abc import Sequence


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
