from pathlib import Path
from unittest.mock import Mock

import pytest

from app.parsers.base import BaseOCREngine, BaseTableExtractor
from app.parsers.pdf import PdfParser


class MockOCREngine(BaseOCREngine):
    def extract_text(self, path: Path, page_number: int) -> str:
        return "OCR Text"


class MockTableExtractor(BaseTableExtractor):
    def extract_tables(self, path: Path, page_number: int) -> str:
        return "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |"


@pytest.fixture
def dummy_pdf_path(tmp_path: Path) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Dense text goes here. " * 5)  # Make it long enough to avoid OCR

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Short")  # Very short text, should trigger OCR

    p = tmp_path / "test.pdf"
    pdf.output(str(p))
    return p


def test_pdf_parser_no_ocr_no_tables(dummy_pdf_path: Path) -> None:
    parser = PdfParser()
    pages = parser.parse(dummy_pdf_path)
    assert len(pages) == 2
    assert "Dense text" in pages[0]
    assert "Short" in pages[1]


def test_pdf_parser_with_tables(dummy_pdf_path: Path) -> None:
    table_ext = MockTableExtractor()
    parser = PdfParser(table_extractor=table_ext)

    pages = parser.parse(dummy_pdf_path)

    assert "| Col1 | Col2 |" in pages[0]
    assert "| Col1 | Col2 |" in pages[1]
    assert "Dense text" in pages[0]


def test_pdf_parser_with_ocr_threshold(dummy_pdf_path: Path, mocker: Mock) -> None:
    ocr_engine = MockOCREngine()
    spy = mocker.spy(ocr_engine, "extract_text")

    parser = PdfParser(ocr_engine=ocr_engine, ocr_threshold=50)
    pages = parser.parse(dummy_pdf_path)

    # Page 1 has dense text (>50 chars), should NOT trigger OCR
    # Page 2 has "Short" (<50 chars), SHOULD trigger OCR

    assert spy.call_count == 1
    assert "OCR Text" in pages[1]
    # In our logic, OCR text should either replace or be appended. Let's assume
    # it replaces if it was empty, or appends.
    # We will test that it contains the OCR text.
