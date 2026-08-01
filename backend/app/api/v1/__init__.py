from fastapi import APIRouter

from .chunk import chunk_router
from .embed import embed_router
from .index import index_router
from .info import info_router
from .parse import parse_router
from .retrieval import retrieval_router
from .upload import upload_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(info_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(parse_router)
api_v1_router.include_router(chunk_router)
api_v1_router.include_router(embed_router)
api_v1_router.include_router(index_router)
api_v1_router.include_router(retrieval_router)

__all__ = ["api_v1_router"]
