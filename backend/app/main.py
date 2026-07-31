from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.v1 import api_v1_router
from .core import EmptyFileError, FileTooLargeError, get_settings

settings = get_settings()


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


@app.exception_handler(EmptyFileError)
async def empty_file_handler(request: Request, exc: EmptyFileError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(
    request: Request, exc: FileTooLargeError
) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Max size is {exc.max_bytes // 1048576} MB."},
    )


app.include_router(api_v1_router)
