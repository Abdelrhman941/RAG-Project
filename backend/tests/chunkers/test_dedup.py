import hashlib
import uuid

from app.chunkers.dedup import is_duplicate
from app.core import SourceType
from app.schemas.chunk import Chunk


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content=content,
        start_char=0,
        end_char=len(content),
        char_count=len(content),
        source_type=SourceType.PDF,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def test_is_duplicate_exact_match() -> None:
    c = _make_chunk("This is an exact boilerplate disclaimer.")
    assert is_duplicate(c, [c], threshold=0.97)


def test_is_duplicate_highly_similar() -> None:
    c1 = _make_chunk("This is a highly similar chunk of text.")
    c2 = _make_chunk("This is a highly similar chunk of texts")  # 1-char diff
    assert is_duplicate(c2, [c1], threshold=0.90)


def test_is_not_duplicate_different_text() -> None:
    c1 = _make_chunk("This is a completely different chunk of text.")
    c2 = _make_chunk("Apples and oranges are fruits.")
    assert not is_duplicate(c2, [c1], threshold=0.90)


def test_is_not_duplicate_empty_history() -> None:
    c = _make_chunk("First chunk ever.")
    assert not is_duplicate(c, [], threshold=0.90)


def test_fast_fail_length_difference() -> None:
    c1 = _make_chunk("Short.")
    c2 = _make_chunk(
        "A very long string that should fail the length check quickly "
        "before running difflib's expensive ratio calculation."
    )
    assert not is_duplicate(c2, [c1], threshold=0.90)
