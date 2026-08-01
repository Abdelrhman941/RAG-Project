from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Interface every embedding provider must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text, in order."""
