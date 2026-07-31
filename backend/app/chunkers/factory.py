from ..core.enums.chunking import ChunkingStrategy
from ..core.exceptions import UnsupportedChunkingStrategyError
from .base import BaseChunker
from .character import CharacterChunker

_CHUNKERS: dict[ChunkingStrategy, BaseChunker] = {
    ChunkingStrategy.CHARACTER: CharacterChunker(),
}


def get_chunker(strategy: ChunkingStrategy) -> BaseChunker:
    try:
        return _CHUNKERS[strategy]
    except KeyError:
        raise UnsupportedChunkingStrategyError(strategy.value) from None
