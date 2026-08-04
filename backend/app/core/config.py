import tomllib
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import (
    DistanceMetric,
    EmbeddingProviderName,
    Environment,
    LLMProviderName,
    VectorStoreProvider,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BYTES_PER_MB = 1024 * 1024


@cache
def get_project_metadata() -> dict[str, str]:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        return cast(dict[str, str], tomllib.load(f)["project"])


_PROJECT_METADATA = get_project_metadata()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------ Application ------------
    APP_NAME: str = _PROJECT_METADATA["name"]
    APP_VERSION: str = _PROJECT_METADATA["version"]
    APP_DESCRIPTION: str = _PROJECT_METADATA["description"]
    ENVIRONMENT: Environment = Environment.LOCAL

    # ------------ Storage ------------
    MAX_FILE_SIZE_MB: int = Field(default=50, gt=0)
    UPLOAD_DIR: Path = _PROJECT_ROOT / "data" / "uploads"

    # ------------ Chunking ------------
    DEFAULT_EMBEDDING_CHUNK_SIZE: int = Field(default=500, gt=0)
    DEFAULT_EMBEDDING_OVERLAP: int = Field(default=50, ge=0)
    DEFAULT_PROMPT_CHUNK_SIZE: int = Field(default=1000, gt=0)
    DEFAULT_PROMPT_OVERLAP: int = Field(default=100, ge=0)
    MAX_PROMPT_CHUNK_SIZE: int = Field(default=3000, gt=0)
    MIN_CHUNK_CHARS: int = Field(default=50, gt=0)
    DEDUP_SIMILARITY_THRESHOLD: float = Field(default=0.97, ge=0.0, le=1.0)

    # ------------ Embedding ------------
    EMBEDDING_PROVIDER: EmbeddingProviderName = (
        EmbeddingProviderName.SENTENCE_TRANSFORMER
    )
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SPARSE_EMBEDDING_MODEL_NAME: str | None = None
    RERANKER_MODEL_NAME: str | None = None
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        gt=0,
    )
    EMBEDDING_NORMALIZE: bool = True

    # ------------ Vector Store ------------
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.QDRANT
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
    QDRANT_GRPC_PORT: int = Field(default=6334, ge=1, le=65535)
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "documents"
    DISTANCE_METRIC: DistanceMetric = DistanceMetric.COSINE

    # ------------ Retrieval ------------
    DEFAULT_TOP_K: int = Field(default=5, gt=0)
    MAX_TOP_K: int = Field(default=20, gt=0)
    MIN_SCORE: float = Field(default=0.1, ge=0.0, le=1.0)
    RETRIEVAL_FETCH_K: int = Field(default=15, gt=0)
    RERANK_MIN_SCORE: float = Field(default=0.3, ge=0.0, le=1.0)

    # ------------ LLM ------------
    LLM_PROVIDER: LLMProviderName = LLMProviderName.GROQ
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0)

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * _BYTES_PER_MB

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.DEFAULT_EMBEDDING_OVERLAP >= self.DEFAULT_EMBEDDING_CHUNK_SIZE:
            raise ValueError(
                "DEFAULT_EMBEDDING_OVERLAP must be smaller than "
                "DEFAULT_EMBEDDING_CHUNK_SIZE."
            )
        return self

    @model_validator(mode="after")
    def validate_vector_store(self) -> "Settings":
        if not self.QDRANT_COLLECTION.strip():
            raise ValueError("QDRANT_COLLECTION must not be empty.")
        return self

    @model_validator(mode="after")
    def validate_retrieval(self) -> "Settings":
        if self.DEFAULT_TOP_K > self.MAX_TOP_K:
            raise ValueError("DEFAULT_TOP_K must be less than or equal to MAX_TOP_K.")
        return self


@cache
def get_settings() -> Settings:
    return Settings()
