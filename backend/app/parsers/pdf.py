from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..core import ParsingError
from .base import BaseParser


class PdfParser(BaseParser):
    def parse(self, path: Path) -> list[str]:
        try:
            reader = PdfReader(path)
            return [page.extract_text() or "" for page in reader.pages]
        except PdfReadError as exc:
            raise ParsingError(f"Could not read '{path.name}' as a valid PDF.") from exc
