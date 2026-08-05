# Sprint 10 Review Checklist

## P0 - Critical Fixes
* [x] Fix the `embed_document` issue (useless computation): If it's for debugging, rename the endpoint to `/documents/{id}/embed/preview` and return a sample. If not needed, delete it entirely since `/index` covers the real use case.
* [x] Fix the `parse_document` file handle leak: The sync generator inside the parser keeps a file handle open. Make `parse_document` an `@asynccontextmanager` to ensure the handle is closed cleanly if the caller stops prematurely.

## P1 - Architectural Improvements
* [x] Convert all services to Classes with Dependency Injection (DI): `GenerationService` is a class with `__init__`, but others are free functions. Convert them all to classes for better testability.
* [x] Fix `RetrievalServiceAdapter` casting: In `api/deps.py`, the adapter is cast without protection. Make `RetrievalServiceAdapter` explicitly inherit from `RetrievalServicePort` and remove the cast.

## P2 - Refactoring & Code Quality
* [x] Consolidate document routers: There are 5 different routers with the `/documents` prefix. Merge them into a single `documents_router` in `api/v1/documents.py`.
* [x] Remove `assert` in production code: In the `retrieval.py` route, `assert request.top_k is not None` will be stripped in production. Use proper type narrowing (`if request.top_k is None: raise HTTPException`) or rely on a `default_factory`.
* [x] Fix function-level imports: Move `settings` imports inside functions in `document_chunker.py`, `document_embedder.py`, and `document_indexer.py` to the top of the file, or pass them as parameters.

## P3 - Testing & Observability
* [x] Add a full pipeline integration test.
* [x] Set up centralized structured logging in `main.py`.
