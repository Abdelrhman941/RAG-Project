from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core import get_settings


class TopKQueryRequest(BaseModel):
    """Shared shape for any endpoint accepting a free-text query + top_k.

    `top_k` defaults and its upper bound are resolved per-instance (via
    the validator below), not at import time — so overriding `Settings`
    in tests actually takes effect.
    """

    model_config = ConfigDict(str_strip_whitespace=True)
    query: Annotated[str, Field(min_length=1)]
    top_k: Annotated[int, Field(ge=1)] | None = Field(
        default=None,
        description="Maximum number of chunks to use. Defaults to Settings.DEFAULT_TOP_K.",  # noqa: E501
    )

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return value

    @model_validator(mode="after")
    def _resolve_top_k(self) -> "TopKQueryRequest":
        settings = get_settings()
        if self.top_k is None:
            self.top_k = settings.DEFAULT_TOP_K
        elif self.top_k > settings.MAX_TOP_K:
            raise ValueError(
                f"top_k must be less than or equal to {settings.MAX_TOP_K}."
            )
        return self
