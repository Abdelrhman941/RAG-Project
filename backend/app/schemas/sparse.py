from pydantic import BaseModel, ConfigDict, model_validator


class SparseVector(BaseModel):
    """Domain model representing a sparse vector for hybrid search."""

    model_config = ConfigDict(frozen=True)

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        return self
