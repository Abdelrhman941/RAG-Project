from sentence_transformers import SentenceTransformer

from ..core import EmbeddingError, get_settings
from .base import BaseEmbeddingProvider

settings = get_settings()


class SentenceTransformerProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=settings.EMBEDDING_DEVICE)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise EmbeddingError(
                f"Model '{self._model_name}' does not expose an embedding dimension."
            )
        return int(dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]
