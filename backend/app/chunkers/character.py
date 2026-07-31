from .base import BaseChunker


class CharacterChunker(BaseChunker):
    """Fixed-size sliding-window chunker with character-based overlap.

    Simple, fast, and deterministic — no tokenizer or model dependency.
    Callers are expected to guarantee chunk_size > 0 and 0 <= overlap <
    chunk_size before calling; this class does not re-validate that.
    """

    def chunk(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        if not text:
            return []

        step = chunk_size - overlap
        text_length = len(text)

        spans: list[tuple[str, int, int]] = []
        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            spans.append((text[start:end], start, end))
            if end == text_length:
                break
            start += step

        return spans
