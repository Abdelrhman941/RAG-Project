import math
from collections.abc import Sequence

from .base import BaseRerankerProvider


class CrossEncoderReranker(BaseRerankerProvider):
    """Cross-Encoder reranker using sentence-transformers."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)

        # Convert to list if it's a numpy array, handle single scalar
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]

        # Apply sigmoid to bound scores to [0, 1] since CrossEncoders output raw logits
        return [1.0 / (1.0 + math.exp(-score)) for score in scores]
