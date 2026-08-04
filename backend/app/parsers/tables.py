import logging
from pathlib import Path

import pdfplumber

from .base import BaseTableExtractor

logger = logging.getLogger(__name__)


class PdfPlumberTableExtractor(BaseTableExtractor):
    """Extracts tables from PDF pages and formats them as Markdown."""

    def extract_tables(self, path: Path, page_number: int) -> str:
        try:
            with pdfplumber.open(path) as pdf:
                # pdfplumber uses 0-based indexing for pages
                if page_number < 1 or page_number > len(pdf.pages):
                    return ""

                page = pdf.pages[page_number - 1]
                if page is None:
                    return ""
                    
                tables = page.extract_tables()

                if not tables:
                    return ""

                markdown_tables = []
                for table in tables:
                    markdown_tables.append(self._to_markdown(table))

                return "\n\n".join(markdown_tables)
        except Exception as exc:
            logger.warning(
                f"Table extraction failed for '{path.name}' page {page_number}: {exc}"
            )
            return ""

    def _to_markdown(self, table: list[list[str | None]]) -> str:
        """Convert a 2D list into a Markdown formatted table."""
        if not table:
            return ""

        # Clean None values and newlines within cells
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Replace newlines so Markdown tables don't break
                    cleaned_row.append(str(cell).replace("\n", " ").strip())
            cleaned_table.append(cleaned_row)

        header = cleaned_table[0]
        md_lines = []

        # Build Header
        md_lines.append("| " + " | ".join(header) + " |")

        # Build Separator
        separator = ["---"] * len(header)
        md_lines.append("| " + " | ".join(separator) + " |")

        # Build Body
        for raw_row in cleaned_table[1:]:
            # Ensure row length matches header length to avoid broken tables
            padded_row = list(raw_row)
            while len(padded_row) < len(header):
                padded_row.append("")
            padded_row = padded_row[: len(header)]
            md_lines.append("| " + " | ".join(padded_row) + " |")

        return "\n".join(md_lines)
