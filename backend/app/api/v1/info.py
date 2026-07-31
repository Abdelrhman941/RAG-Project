from fastapi import APIRouter

from ..deps import SettingsDep

info_router = APIRouter(prefix="", tags=["Health"])


@info_router.get("/")
def root(settings: SettingsDep) -> dict[str, str]:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
    }


@info_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
