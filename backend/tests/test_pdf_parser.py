from collections.abc import Callable
from pathlib import Path

import pytest

from app.core.exceptions import ParsingError
from app.parsers.pdf import PdfParser


@pytest.fixture()
def parser() -> PdfParser:
    return PdfParser()


def test_parse_single_page_extracts_text(
    parser: PdfParser, tmp_path: Path, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    file_path = tmp_path / "single.pdf"
    file_path.write_bytes(make_pdf_bytes(["Hello from page one"]))

    pages = parser.parse(file_path)

    assert len(pages) == 1
    assert "Hello from page one" in pages[0]


def test_parse_multi_page_returns_correct_page_count_and_order(
    parser: PdfParser, tmp_path: Path, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    file_path = tmp_path / "multi.pdf"
    file_path.write_bytes(
        make_pdf_bytes(
            ["First page content", "Second page content", "Third page content"]
        )
    )

    pages = parser.parse(file_path)

    assert len(pages) == 3
    assert "First page content" in pages[0]
    assert "Second page content" in pages[1]
    assert "Third page content" in pages[2]


def test_parse_blank_page_returns_empty_string_not_none(
    parser: PdfParser, tmp_path: Path, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    file_path = tmp_path / "blank.pdf"
    file_path.write_bytes(make_pdf_bytes([""]))

    pages = parser.parse(file_path)

    assert pages == [""]


def test_parse_returns_list_of_strings(
    parser: PdfParser, tmp_path: Path, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    file_path = tmp_path / "typed.pdf"
    file_path.write_bytes(make_pdf_bytes(["some text"]))

    pages = parser.parse(file_path)

    assert isinstance(pages, list)
    assert all(isinstance(page, str) for page in pages)


def test_parse_corrupted_pdf_raises_parsing_error(
    parser: PdfParser, tmp_path: Path
) -> None:
    file_path = tmp_path / "corrupted.pdf"
    file_path.write_bytes(b"this is not a valid pdf file at all")

    with pytest.raises(ParsingError, match="Could not read"):
        parser.parse(file_path)
