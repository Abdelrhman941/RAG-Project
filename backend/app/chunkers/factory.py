from ..core import ChunkingStrategy, UnsupportedChunkingStrategyError
from .base import BaseChunker
from .token import RecursiveChunker

_CHUNKERS: dict[ChunkingStrategy, BaseChunker] = {
    ChunkingStrategy.TOKEN: RecursiveChunker(),
}


def get_chunker(strategy: ChunkingStrategy) -> BaseChunker:
    try:
        return _CHUNKERS[strategy]
    except KeyError:
        raise UnsupportedChunkingStrategyError(strategy) from None
