import pytest

from app.chunkers.character import CharacterChunker


@pytest.fixture()
def chunker() -> CharacterChunker:
    return CharacterChunker()


def test_empty_text_returns_no_chunks(chunker: CharacterChunker) -> None:
    spans = chunker.chunk("", chunk_size=100, overlap=20)

    assert spans == []


def test_text_shorter_than_chunk_size_returns_single_chunk(
    chunker: CharacterChunker,
) -> None:
    text = "short text"

    spans = chunker.chunk(text, chunk_size=1000, overlap=200)

    assert spans == [(text, 0, len(text))]


def test_text_exactly_chunk_size_returns_single_chunk(
    chunker: CharacterChunker,
) -> None:
    text = "x" * 500

    spans = chunker.chunk(text, chunk_size=500, overlap=100)

    assert spans == [(text, 0, 500)]


def test_no_overlap_splits_into_exact_windows(chunker: CharacterChunker) -> None:
    text = "x" * 1000

    spans = chunker.chunk(text, chunk_size=250, overlap=0)

    assert [start for _, start, _ in spans] == [0, 250, 500, 750]
    assert [end for _, _, end in spans] == [250, 500, 750, 1000]
    assert all(len(content) == 250 for content, _, _ in spans)


def test_overlap_produces_overlapping_windows(chunker: CharacterChunker) -> None:
    text = "0123456789" * 10  # 100 chars

    spans = chunker.chunk(text, chunk_size=30, overlap=10)

    # step = chunk_size - overlap = 20
    starts = [start for _, start, _ in spans]
    assert starts == [0, 20, 40, 60, 80]
    # every window after the first should share `overlap` chars with the previous one
    for i in range(1, len(spans)):
        _prev_content, _prev_start, prev_end = spans[i - 1]
        _curr_content, curr_start, _curr_end = spans[i]
        assert curr_start == prev_end - 10


def test_last_chunk_is_not_padded_and_stops_at_text_end(
    chunker: CharacterChunker,
) -> None:
    text = "x" * 1050  # not a clean multiple of (chunk_size - overlap)

    spans = chunker.chunk(text, chunk_size=500, overlap=100)

    last_content, last_start, last_end = spans[-1]
    assert last_end == len(text)
    assert last_content == text[last_start:]
    # no span should ever run past the text length
    assert all(end <= len(text) for _, _, end in spans)


def test_spans_reconstruct_original_text_positions(chunker: CharacterChunker) -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20

    spans = chunker.chunk(text, chunk_size=100, overlap=20)

    for content, start, end in spans:
        assert text[start:end] == content
        assert len(content) == end - start


def test_full_text_is_covered_with_no_gaps(chunker: CharacterChunker) -> None:
    text = "y" * 733

    spans = chunker.chunk(text, chunk_size=200, overlap=50)

    # every character index must be covered by at least one span
    covered: set[int] = set()
    for _, start, end in spans:
        covered.update(range(start, end))

    assert covered == set(range(len(text)))
