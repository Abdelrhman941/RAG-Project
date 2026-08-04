from app.core.config import Settings


def test_min_chunk_chars_has_sensible_default() -> None:
    s = Settings()
    assert s.MIN_CHUNK_CHARS == 50
    assert s.MIN_CHUNK_CHARS > 0


def test_dedup_similarity_threshold_default() -> None:
    s = Settings()
    assert s.DEDUP_SIMILARITY_THRESHOLD == 0.97
    assert 0.0 < s.DEDUP_SIMILARITY_THRESHOLD <= 1.0


def test_chunk_size_defaults() -> None:
    s = Settings()
    assert s.DEFAULT_EMBEDDING_CHUNK_SIZE == 500
    assert s.DEFAULT_EMBEDDING_OVERLAP == 50
