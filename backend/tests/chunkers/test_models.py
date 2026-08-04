import pytest

from app.chunkers.models import ChunkingConfig, ChunkSpan
from app.core import ChunkingStrategy, SourceType


def test_chunking_config_accepts_min_chunk_chars() -> None:
    cfg = ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=512,
        embedding_overlap=50,
        min_chunk_chars=10,
        source_type=SourceType.PDF,
    )
    assert cfg.min_chunk_chars == 10
    assert cfg.source_type == SourceType.PDF


def test_chunking_config_min_chunk_chars_must_be_positive() -> None:
    from app.core.exceptions import InvalidChunkingParametersError

    with pytest.raises(InvalidChunkingParametersError):
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN,
            embedding_chunk_size=512,
            embedding_overlap=0,
            min_chunk_chars=0,
            source_type=SourceType.TXT,
        )


def test_chunk_span_carries_source_type() -> None:
    span = ChunkSpan(
        content="hello", start_char=0, end_char=5, source_type=SourceType.MD
    )
    assert span.source_type == SourceType.MD
