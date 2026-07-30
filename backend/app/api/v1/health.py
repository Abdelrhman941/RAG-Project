from fastapi import APIRouter

from ..deps import SettingsDep

health_router = APIRouter(prefix="", tags=["Health"])


@health_router.get("/")
def root(settings: SettingsDep) -> dict[str, str]:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
    }


@health_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
