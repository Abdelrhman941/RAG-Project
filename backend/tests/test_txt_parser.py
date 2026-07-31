from pathlib import Path

import pytest

from app.core.exceptions import ParsingError
from app.parsers.txt import TxtParser


@pytest.fixture()
def parser() -> TxtParser:
    return TxtParser()


def test_parse_returns_single_page_with_full_content(
    parser: TxtParser, tmp_path: Path
) -> None:
    content = "Hello, this is a test document.\nWith a second line."
    file_path = tmp_path / "notes.txt"
    file_path.write_text(content, encoding="utf-8")

    pages = parser.parse(file_path)

    assert pages == [content]


def test_parse_returns_list_of_length_one(parser: TxtParser, tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("irrelevant content", encoding="utf-8")

    pages = parser.parse(file_path)

    assert isinstance(pages, list)
    assert len(pages) == 1


def test_parse_empty_file_returns_single_empty_string(
    parser: TxtParser, tmp_path: Path
) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    pages = parser.parse(file_path)

    assert pages == [""]


def test_parse_preserves_unicode_content(parser: TxtParser, tmp_path: Path) -> None:
    content = "مرحبا بالعالم - hello world - 你好"
    file_path = tmp_path / "unicode.txt"
    file_path.write_text(content, encoding="utf-8")

    pages = parser.parse(file_path)

    assert pages == [content]


def test_parse_invalid_utf8_raises_parsing_error(
    parser: TxtParser, tmp_path: Path
) -> None:
    file_path = tmp_path / "bad_encoding.txt"
    # 0xff 0xfe is not valid UTF-8
    file_path.write_bytes(b"\xff\xfe invalid bytes")

    with pytest.raises(ParsingError, match="Could not decode"):
        parser.parse(file_path)


def test_md_parser_behaves_like_txt_parser(tmp_path: Path) -> None:
    from app.parsers.md import MdParser

    content = "# Heading\n\nSome **markdown** content."
    file_path = tmp_path / "readme.md"
    file_path.write_text(content, encoding="utf-8")

    pages = MdParser().parse(file_path)

    assert pages == [content]
