from fastapi import APIRouter

from .info import health_router
from .upload import upload_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(upload_router)

__all__ = ["api_v1_router"]
