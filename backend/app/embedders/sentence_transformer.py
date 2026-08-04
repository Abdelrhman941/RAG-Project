from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from ..core import EmbeddingError
from .base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """SentenceTransformers embedding provider."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        normalize_embeddings: bool,
        batch_size: int,
    ) -> None:
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        try:
            self._model = SentenceTransformer(
                model_name,
                device=device,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}'."
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise EmbeddingError(
                f"Model '{self._model_name}' does not expose an embedding dimension."
            )
        return int(dimension)

    @property
    def max_sequence_length(self) -> int:
        seq_len = self._model.max_seq_length
        if seq_len is None:
            raise EmbeddingError(
                f"Model '{self._model_name}' does not expose max_seq_length."
            )
        return int(seq_len)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        from typing import cast

        if not texts:
            return []
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )
            return cast(list[list[float]], vectors.tolist())
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate document embeddings using '{self._model_name}'."
            ) from exc

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Query text must not be empty.")
        return self.embed_documents([text])[0]
