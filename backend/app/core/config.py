from functools import cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums.embedding import EmbeddingProviderName
from .enums.vector_store import DistanceMetric, VectorStoreProvider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", env_file_encoding="utf-8"
    )

    # ----- Application -----
    APP_NAME: str = "RAG"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "A RAG application"
    ENVIRONMENT: str = "local"
    EMBEDDING_DEVICE: str = "cpu"

    # ----- File Upload -----
    MAX_FILE_SIZE_MB: int = 50
    UPLOAD_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "uploads"

    # ----- Chunking -----
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200

    # ----- Embedding -----
    EMBEDDING_PROVIDER: EmbeddingProviderName = (
        EmbeddingProviderName.SENTENCE_TRANSFORMER
    )
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

    # ----- Vector Store -----
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.QDRANT
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "documents"
    DISTANCE_METRIC: DistanceMetric = DistanceMetric.COSINE

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1048576

    @model_validator(mode="after")
    def validate_default_chunking_parameters(self) -> "Settings":
        if self.DEFAULT_CHUNK_SIZE <= 0:
            raise ValueError("DEFAULT_CHUNK_SIZE must be greater than 0.")
        if self.DEFAULT_CHUNK_OVERLAP < 0:
            raise ValueError(
                "DEFAULT_CHUNK_OVERLAP must be greater than or equal to 0."
            )
        if self.DEFAULT_CHUNK_OVERLAP >= self.DEFAULT_CHUNK_SIZE:
            raise ValueError(
                "DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE."
            )
        return self

    @model_validator(mode="after")
    def validate_vector_store_settings(self) -> "Settings":
        if not self.QDRANT_COLLECTION.strip():
            raise ValueError("QDRANT_COLLECTION must not be empty.")
        if not (1 <= self.QDRANT_PORT <= 65535):
            raise ValueError("QDRANT_PORT must be a valid TCP port.")
        return self


@cache
def get_settings() -> Settings:
    return Settings()
