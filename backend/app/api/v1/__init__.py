from fastapi import APIRouter

from .info import info_router
from .parse import parse_router
from .upload import upload_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(info_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(parse_router)

__all__ = ["api_v1_router"]
