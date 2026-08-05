from fastapi import APIRouter

from .chat import chat_router
from .documents import documents_router
from .info import info_router
from .retrieval import retrieval_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(info_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(chat_router)

__all__ = ["api_v1_router"]
