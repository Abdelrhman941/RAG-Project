from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from app.embedders.cross_encoder import CrossEncoderReranker


@pytest.fixture
def mock_cross_encoder() -> Generator[MagicMock, None, None]:
    with patch("sentence_transformers.CrossEncoder", autospec=True) as mock_ce:
        instance = mock_ce.return_value
        yield instance


def test_cross_encoder_rerank(mock_cross_encoder: MagicMock) -> None:
    # Mock scores (e.g. from logits)
    mock_cross_encoder.predict.return_value = [0.1, -1.0, 3.5]

    reranker = CrossEncoderReranker(model_name="test-model")

    query = "test query"
    texts = ["text1", "text2", "text3"]
    scores = reranker.rerank(query, texts)

    # Check predict was called properly
    mock_cross_encoder.predict.assert_called_once()
    args = mock_cross_encoder.predict.call_args[0][0]
    assert args == [
        ("test query", "text1"),
        ("test query", "text2"),
        ("test query", "text3"),
    ]

    # Assert sigmoid activation bounding (assuming we use sigmoid)
    # The values depend on the exact implementation, but should be in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in scores)
