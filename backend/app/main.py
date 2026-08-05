from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.v1 import api_v1_router
from .core import DocumentError, get_settings, setup_logging

settings = get_settings()
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
)


@app.exception_handler(DocumentError)
async def document_error_handler(request: Request, exc: DocumentError) -> JSONResponse:
    """Single handler for the entire custom exception hierarchy.

    Every `DocumentError` subclass carries its own `status_code`
    (see `core/exceptions.py`), so this one handler covers all of
    them — including any future subclass, with zero changes here.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


app.include_router(api_v1_router)
