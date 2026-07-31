from functools import cache
from pathlib import Path

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

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1048576


@cache
def get_settings() -> Settings:
    return Settings()
