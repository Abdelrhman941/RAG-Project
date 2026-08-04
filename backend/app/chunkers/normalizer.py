import re


def normalize_text(text: str) -> str:
    """Normalize text before chunking."""
    if not text:
        return text
    # 1. Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. Remove null bytes and control chars (except \n and \t)
    text = "".join(c for c in text if c.isprintable() or c in ("\n", "\t"))
    # 3. Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
