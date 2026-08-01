from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api.v1 import api_v1_router
from .core import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    EmbeddingError,
    EmptyFileError,
    FileTooLargeError,
    IndexingError,
    InvalidChunkingParametersError,
    ParsingError,
    RetrievalError,
    UnsupportedChunkingStrategyError,
    UnsupportedDocumentTypeError,
    UnsupportedEmbeddingProviderError,
    UnsupportedVectorStoreProviderError,
    VectorDimensionMismatchError,
    VectorStoreUnavailableError,
    get_settings,
)

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


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(UnsupportedDocumentTypeError)
async def unsupported_type_handler(
    request: Request, exc: UnsupportedDocumentTypeError
) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(ParsingError)
async def parsing_error_handler(request: Request, exc: ParsingError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(InvalidChunkingParametersError)
async def invalid_chunking_parameters_handler(
    request: Request, exc: InvalidChunkingParametersError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(UnsupportedChunkingStrategyError)
async def unsupported_chunking_strategy_handler(
    request: Request, exc: UnsupportedChunkingStrategyError
) -> JSONResponse:
    return JSONResponse(status_code=501, content={"detail": str(exc)})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(
    request: Request, exc: EmbeddingError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(UnsupportedEmbeddingProviderError)
async def unsupported_embedding_provider_handler(
    request: Request, exc: UnsupportedEmbeddingProviderError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(VectorStoreUnavailableError)
async def vector_store_unavailable_handler(
    request: Request, exc: VectorStoreUnavailableError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(VectorDimensionMismatchError)
async def vector_dimension_mismatch_handler(
    request: Request, exc: VectorDimensionMismatchError
) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(IndexingError)
async def indexing_error_handler(request: Request, exc: IndexingError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(UnsupportedVectorStoreProviderError)
async def unsupported_vector_store_provider_handler(
    request: Request, exc: UnsupportedVectorStoreProviderError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------- Retrieval ----------
@app.exception_handler(CollectionNotFoundError)
async def collection_not_found_handler(
    request: Request, exc: CollectionNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(
    request: Request, exc: RetrievalError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(api_v1_router)
