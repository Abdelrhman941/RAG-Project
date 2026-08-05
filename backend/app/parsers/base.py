from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, path: Path) -> Generator[str, None, None]:
        """Parse a document into ordered text segments (yields pages/sections)."""
        raise NotImplementedError


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(self, path: Path, page_number: int) -> str:
        """Extract text from a specific page using OCR."""
        raise NotImplementedError


class BaseTableExtractor(ABC):
    """Abstract base class for Table extraction engines."""

    @abstractmethod
    def extract_tables(self, path: Path, page_number: int) -> str:
        """Extract tables from a specific page and format as Markdown."""
        raise NotImplementedError
