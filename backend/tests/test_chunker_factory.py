import pytest

from app.chunkers.character import CharacterChunker
from app.chunkers.factory import _CHUNKERS, get_chunker
from app.core.enums.chunking import ChunkingStrategy
from app.core.exceptions import UnsupportedChunkingStrategyError


def test_get_chunker_returns_character_chunker_for_character_strategy() -> None:
    chunker = get_chunker(ChunkingStrategy.CHARACTER)

    assert isinstance(chunker, CharacterChunker)


def test_get_chunker_returns_singleton_instance() -> None:
    first_call = get_chunker(ChunkingStrategy.CHARACTER)
    second_call = get_chunker(ChunkingStrategy.CHARACTER)

    assert first_call is second_call


def test_get_chunker_covers_every_declared_strategy() -> None:
    for strategy in ChunkingStrategy:
        # Should not raise for any strategy currently declared in the enum.
        assert get_chunker(strategy) is not None


def test_get_chunker_raises_for_unregistered_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(_CHUNKERS, ChunkingStrategy.CHARACTER)

    with pytest.raises(UnsupportedChunkingStrategyError) as exc_info:
        get_chunker(ChunkingStrategy.CHARACTER)

    assert exc_info.value.strategy == ChunkingStrategy.CHARACTER.value
