from enum import Enum


class Environment(str, Enum):
    """Application runtime environment."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
