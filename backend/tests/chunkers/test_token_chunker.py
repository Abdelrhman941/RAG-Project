from app.chunkers.models import ChunkingConfig
from app.chunkers.token import RecursiveChunker
from app.core import ChunkingStrategy, SourceType

PDF = SourceType.PDF


def _cfg(
    embedding_chunk_size: int = 100,
    embedding_overlap: int = 10,
    min_chunk_chars: int = 20,
) -> ChunkingConfig:
    return ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=embedding_chunk_size,
        embedding_overlap=embedding_overlap,
        min_chunk_chars=min_chunk_chars,
        source_type=PDF,
    )


def test_empty_text_returns_no_spans() -> None:
    chunker = RecursiveChunker()
    assert list(chunker.chunk("", _cfg())) == []


def test_whitespace_only_returns_no_spans() -> None:
    chunker = RecursiveChunker()
    assert list(chunker.chunk("   \n\t  ", _cfg())) == []


def test_normal_text_returns_spans_with_correct_source_type() -> None:
    chunker = RecursiveChunker()
    text = "word " * 200
    spans = chunker.chunk(
        text, _cfg(embedding_chunk_size=50, embedding_overlap=5, min_chunk_chars=10)
    )
    assert all(s.source_type == PDF for s in spans)


def test_hierarchical_chunking_creates_parent_references() -> None:
    chunker = RecursiveChunker()
    text = "word " * 600
    # prompt_chunk_size=200, embedding_chunk_size=50
    # Expect multiple parents, each with multiple children
    cfg = ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=50,
        embedding_overlap=5,
        prompt_chunk_size=200,
        prompt_overlap=20,
        min_chunk_chars=10,
        source_type=SourceType.TXT,
    )
    spans = list(chunker.chunk(text, cfg))

    assert len(spans) > 0
    # Check that all spans have parent fields populated
    assert all(s.parent_chunk_id is not None for s in spans)
    assert all(s.parent_content is not None for s in spans)

    # Check that children share the same parent_chunk_id if they came from
    # the same parent.
    parent_ids = {s.parent_chunk_id for s in spans}
    assert len(parent_ids) > 1

    # Verify child offsets align with original text
    for span in spans:
        assert text[span.start_char : span.end_char] == span.content
        assert span.parent_content is not None and span.parent_content in text


def test_tiny_chunk_is_merged_into_neighbor() -> None:
    chunker = RecursiveChunker()
    long_part = "word " * 60
    short_tail = "end"
    text = long_part + short_tail
    spans = chunker.chunk(
        text, _cfg(embedding_chunk_size=200, embedding_overlap=0, min_chunk_chars=20)
    )
    contents = [s.content for s in spans]
    assert not any(c.strip() == "end" for c in contents)


def test_sole_tiny_chunk_is_preserved() -> None:
    chunker = RecursiveChunker()
    text = "tiny abstract."
    spans = list(
        chunker.chunk(
            text,
            _cfg(embedding_chunk_size=200, embedding_overlap=0, min_chunk_chars=50),
        )
    )
    assert len(spans) == 1
    assert spans[0].content == "tiny abstract."


def test_span_offsets_are_within_original_text_bounds() -> None:
    chunker = RecursiveChunker()
    text = "Hello world. " * 40
    cfg = _cfg(embedding_chunk_size=50, embedding_overlap=5, min_chunk_chars=10)
    spans = chunker.chunk(text, cfg)
    for span in spans:
        assert span.start_char >= 0
        assert span.end_char <= len(text)
        assert span.start_char < span.end_char


def test_crlf_in_input_is_normalized() -> None:
    chunker = RecursiveChunker()
    text = "Line one.\r\nLine two.\r\nLine three."
    spans = chunker.chunk(text, _cfg())
    combined = " ".join(s.content for s in spans)
    assert "\r" not in combined
