from collections.abc import Sequence
from typing import cast

from .base import BaseRerankerProvider


class CrossEncoderReranker(BaseRerankerProvider):
    """Cross-Encoder reranker using sentence-transformers."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self._model_name = model_name
        self._model = CrossEncoder(
            model_name, default_activation_function=torch.nn.Sigmoid()
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        pairs = [(query, text) for text in texts]

        # Batch predict gives us the bounded scores directly
        # because we set default_activation_function=torch.nn.Sigmoid()
        raw_scores = self._model.predict(pairs)

        return cast(list[float], raw_scores.tolist())
