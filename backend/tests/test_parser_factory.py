import pytest

from app.core.enums.document import DocumentExtension
from app.core.exceptions import UnsupportedDocumentTypeError
from app.parsers.factory import _PARSERS, get_parser
from app.parsers.md import MdParser
from app.parsers.pdf import PdfParser
from app.parsers.txt import TxtParser


@pytest.mark.parametrize(
    "extension, expected_type",
    [
        (DocumentExtension.PDF, PdfParser),
        (DocumentExtension.TXT, TxtParser),
        (DocumentExtension.MD, MdParser),
    ],
)
def test_get_parser_returns_correct_parser_type(
    extension: DocumentExtension, expected_type: type
) -> None:
    parser = get_parser(extension)

    assert isinstance(parser, expected_type)


def test_get_parser_returns_singleton_instance() -> None:
    first_call = get_parser(DocumentExtension.TXT)
    second_call = get_parser(DocumentExtension.TXT)

    assert first_call is second_call


def test_get_parser_covers_every_document_extension() -> None:
    for extension in DocumentExtension:
        # Should not raise for any officially supported extension.
        assert get_parser(extension) is not None


def test_get_parser_raises_for_unregistered_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(_PARSERS, DocumentExtension.TXT)

    with pytest.raises(UnsupportedDocumentTypeError) as exc_info:
        get_parser(DocumentExtension.TXT)

    assert exc_info.value.extension == DocumentExtension.TXT.value
