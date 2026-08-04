from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..core import ParsingError
from .base import BaseOCREngine, BaseParser, BaseTableExtractor


class PdfParser(BaseParser):
    """Parse PDF pages into page-level text
    segments with optional OCR and Table extraction."""

    def __init__(
        self,
        ocr_engine: BaseOCREngine | None = None,
        table_extractor: BaseTableExtractor | None = None,
        ocr_threshold: int = 50,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._table_extractor = table_extractor
        self._ocr_threshold = ocr_threshold

    def parse(self, path: Path) -> Iterator[str]:
        try:
            reader = PdfReader(path)
        except (OSError, PdfReadError) as exc:
            raise ParsingError(f"Could not read '{path.name}' as a valid PDF.") from exc

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise ParsingError(
                    f"Could not extract text from page {page_number} of '{path.name}'."
                ) from exc

            text = text.replace("\r\n", "\n").replace("\r", "\n")

            # Lazy OCR Evaluation
            if self._ocr_engine and len(text.strip()) < self._ocr_threshold:
                try:
                    ocr_text = self._ocr_engine.extract_text(path, page_number)
                    if ocr_text:
                        text = text + "\n\n" + ocr_text if text.strip() else ocr_text
                except Exception:
                    # Log error but don't fail the entire parsing
                    pass

            if self._table_extractor:
                try:
                    tables_md = self._table_extractor.extract_tables(path, page_number)
                    if tables_md:
                        text = text + "\n\n" + tables_md if text.strip() else tables_md
                except Exception:
                    pass

            yield text
