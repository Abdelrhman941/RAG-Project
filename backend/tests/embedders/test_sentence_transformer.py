from unittest.mock import MagicMock, patch

import pytest

from app.core import EmbeddingError
from app.embedders.sentence_transformer import SentenceTransformerProvider


def test_max_sequence_length_success() -> None:
    mock_model = MagicMock()
    mock_model.max_seq_length = 512

    with patch(
        "app.embedders.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        provider = SentenceTransformerProvider(
            model_name="fake-model",
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )
        assert provider.max_sequence_length == 512


def test_max_sequence_length_missing_raises_error() -> None:
    mock_model = MagicMock()
    mock_model.max_seq_length = None

    with patch(
        "app.embedders.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        provider = SentenceTransformerProvider(
            model_name="fake-model",
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )
        with pytest.raises(EmbeddingError, match="does not expose max_seq_length"):
            _ = provider.max_sequence_length
