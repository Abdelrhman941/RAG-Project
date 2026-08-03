from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, path: Path) -> list[str]:
        """Parse a document into ordered text segments."""
        raise NotImplementedError
