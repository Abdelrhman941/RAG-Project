from collections.abc import Sequence

from fastembed import SparseTextEmbedding

from ..schemas import SparseVector
from .base import BaseSparseEmbeddingProvider


class FastEmbedSparseProvider(BaseSparseEmbeddingProvider):
    """SPLADE sparse embedding provider using FastEmbed."""

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self._model_name = model_name
        # Note: In a production app, models should be loaded lazily or centrally managed
        self._model = SparseTextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_sparse_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        # FastEmbed returns a generator of SparseEmbedding objects
        embeddings = list(self._model.embed(texts))
        return [
            SparseVector(
                indices=list(emb.indices),
                values=list(emb.values),
            )
            for emb in embeddings
        ]

    def embed_sparse_query(self, text: str) -> SparseVector:
        embeddings = list(self._model.query_embed(text))
        emb = embeddings[0]
        return SparseVector(
            indices=list(emb.indices),
            values=list(emb.values),
        )
