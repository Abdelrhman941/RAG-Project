from ..core.enums.document import DocumentExtension
from ..core.exceptions import UnsupportedDocumentTypeError
from .base import BaseParser
from .md import MdParser
from .pdf import PdfParser
from .txt import TxtParser

_PARSERS: dict[DocumentExtension, BaseParser] = {
    DocumentExtension.PDF: PdfParser(),
    DocumentExtension.TXT: TxtParser(),
    DocumentExtension.MD: MdParser(),
}


def get_parser(extension: DocumentExtension) -> BaseParser:
    try:
        return _PARSERS[extension]
    except KeyError:
        raise UnsupportedDocumentTypeError(extension.value) from None
