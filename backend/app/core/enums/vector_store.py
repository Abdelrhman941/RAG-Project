from enum import Enum


class VectorStoreProvider(str, Enum):
    """Vector store provider names."""

    QDRANT = "qdrant"


class DistanceMetric(str, Enum):
    """Distance metric used when creating a vector collection."""

    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"
