import re
from pathlib import Path

from .txt import TxtParser

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


class MdParser(TxtParser):
    """Parse Markdown into heading-preserving sections."""

    def parse(self, path: Path) -> list[str]:
        text = super().parse(path)[0]
        return self._split_into_sections(text)

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        if not normalized.strip():
            return [normalized]
        sections: list[str] = []
        current_lines: list[str] = []
        in_code_block = False

        for line in normalized.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
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
        return sections or [normalized.strip()]
