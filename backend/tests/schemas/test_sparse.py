import pytest
from pydantic import ValidationError

from app.schemas.sparse import SparseVector


def test_sparse_vector_valid() -> None:
    sv = SparseVector(indices=[1, 2], values=[0.5, 0.8])
    assert sv.indices == [1, 2]
    assert sv.values == [0.5, 0.8]


def test_sparse_vector_invalid_lengths() -> None:
    with pytest.raises(
        ValidationError, match="indices and values must have the same length"
    ):
        SparseVector(indices=[1, 2], values=[0.5])
