from functools import cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", env_file_encoding="utf-8"
    )

    # ----- Application -----
    APP_NAME: str = "RAG"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "A RAG application"
    ENVIRONMENT: str = "local"

    # ----- File Upload -----
    MAX_FILE_SIZE_MB: int = 50
    UPLOAD_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "uploads"

    # ----- Chunking -----
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200

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


@cache
def get_settings() -> Settings:
    return Settings()
