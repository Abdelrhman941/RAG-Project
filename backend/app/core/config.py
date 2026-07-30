from functools import lru_cache
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
