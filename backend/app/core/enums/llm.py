from enum import Enum


class LLMProviderName(str, Enum):
    """LLM provider names."""

    GROQ = "groq"
