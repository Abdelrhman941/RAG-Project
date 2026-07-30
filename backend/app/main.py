from fastapi import FastAPI

from .api.v1 import api_v1_router
from .core import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
)

app.include_router(api_v1_router)
