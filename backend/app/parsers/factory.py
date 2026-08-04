from ..core import DocumentExtension, UnsupportedDocumentTypeError
from .base import BaseParser
from .md import MdParser
from .ocr import TesseractOCREngine
from .pdf import PdfParser
from .tables import PdfPlumberTableExtractor
from .txt import TxtParser

_PARSER_REGISTRY: dict[DocumentExtension, BaseParser] = {
    DocumentExtension.PDF: PdfParser(
        ocr_engine=TesseractOCREngine(),
        table_extractor=PdfPlumberTableExtractor(),
        ocr_threshold=50,
    ),
    DocumentExtension.TXT: TxtParser(),
    DocumentExtension.MD: MdParser(),
}


def get_parser(extension: DocumentExtension) -> BaseParser:
    try:
        return _PARSER_REGISTRY[extension]
    except KeyError:
        raise UnsupportedDocumentTypeError(extension.value) from None
