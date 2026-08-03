from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..core import ParsingError
from .base import BaseParser


class PdfParser(BaseParser):
    """Parse PDF pages into page-level text segments."""

    def parse(self, path: Path) -> list[str]:
        try:
            reader = PdfReader(path)
        except (OSError, PdfReadError) as exc:
            raise ParsingError(f"Could not read '{path.name}' as a valid PDF.") from exc

        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise ParsingError(
                    f"Could not extract text from page {page_number} of '{path.name}'."
                ) from exc
            pages.append(text.replace("\r\n", "\n").replace("\r", "\n"))
        return pages
