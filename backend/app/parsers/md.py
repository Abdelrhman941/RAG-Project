import re
from collections.abc import Iterator
from pathlib import Path

from .txt import TxtParser

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


class MdParser(TxtParser):
    """Parse Markdown into heading-preserving sections."""

    def parse(self, path: Path) -> Iterator[str]:
        text = self._read_text(path)
        yield from self._split_into_sections(text)

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        if not text.strip():
            return [text]
        sections: list[str] = []
        current_lines: list[str] = []
        in_code_block = False

        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                in_code_block = not in_code_block
            is_heading = not in_code_block and bool(_HEADING_RE.match(stripped))

            if is_heading and current_lines:
                section = "\n".join(current_lines).strip()
                if section:
                    sections.append(section)
                current_lines = [line]
                continue
            current_lines.append(line)

        if current_lines:
            section = "\n".join(current_lines).strip()
            if section:
                sections.append(section)
        return sections or [text.strip()]
