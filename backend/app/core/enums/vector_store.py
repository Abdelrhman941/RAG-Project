from enum import Enum


class VectorStoreProvider(str, Enum):
    # ----- Supported providers -----
    QDRANT = "qdrant"
    # PGVECTOR = "pgvector"
    # MILVUS = "milvus"


class DistanceMetric(str, Enum):
    """Distance metric used when creating a vector collection.

    Values match Qdrant's `Distance` enum on purpose so the mapping to
    the SDK stays a one-liner inside `QdrantVectorStore` — but this enum
    itself is provider-agnostic and lives in the domain layer.
    """

    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"
