from pathlib import Path

from ..core.exceptions import ParsingError
from .base import BaseParser


class TxtParser(BaseParser):
    def parse(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ParsingError(
                f"Could not decode '{path.name}' as UTF-8 text."
            ) from exc
        return [text]
