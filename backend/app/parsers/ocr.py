import logging
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class TesseractOCREngine(BaseOCREngine):
    """OCR Engine using Tesseract via pdf2image and pytesseract."""

    def extract_text(self, path: Path, page_number: int) -> str:
        try:
            # pdf2image uses 1-based indexing for first_page and last_page
            images = convert_from_path(
                str(path),
                first_page=page_number,
                last_page=page_number,
                dpi=300,
            )
            if not images:
                return ""

            # Extract text from the specific page image
            text = str(pytesseract.image_to_string(images[0]))
            return text.strip()
        except Exception as exc:
            logger.warning(
                f"OCR extraction failed for '{path.name}' page {page_number}: {exc}"
            )
            return ""
