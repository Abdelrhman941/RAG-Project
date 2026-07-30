from fastapi import FastAPI

from .core import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
