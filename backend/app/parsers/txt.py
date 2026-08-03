from pathlib import Path

from charset_normalizer import from_path

from ..core import ParsingError
from .base import BaseParser


class TxtParser(BaseParser):
    """Parse plain text files with UTF-8 first, then encoding detection."""

    def parse(self, path: Path) -> list[str]:
        try:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as err:
                match = from_path(path).best()
                if match is None:
                    raise ParsingError(
                        f"Could not decode '{path.name}' using "
                        "UTF-8 or an auto-detected encoding."
                    ) from err
                text = str(match)
        except OSError as exc:
            raise ParsingError(f"Could not read text file '{path.name}'.") from exc

        return [text.replace("\r\n", "\n").replace("\r", "\n")]
