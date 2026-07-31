from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """Interface every document parser must implement."""

    @abstractmethod
    def parse(self, path: Path) -> list[str]:
        """Parse the file at `path` and return a list of page texts."""
