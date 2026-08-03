from ..core import DocumentExtension, UnsupportedDocumentTypeError
from .base import BaseParser
from .md import MdParser
from .pdf import PdfParser
from .txt import TxtParser

_PARSER_REGISTRY: dict[DocumentExtension, BaseParser] = {
    DocumentExtension.PDF: PdfParser(),
    DocumentExtension.TXT: TxtParser(),
    DocumentExtension.MD: MdParser(),
}


def get_parser(extension: DocumentExtension) -> BaseParser:
    try:
        return _PARSER_REGISTRY[extension]
    except KeyError:
        raise UnsupportedDocumentTypeError(extension.value) from None
