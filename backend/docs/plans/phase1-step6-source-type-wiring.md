# Phase 1: Step 6 Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the hardcoded `source_type` in the chunking service and eliminate duplicated compute (I/O and extension inference) by making `parse_document` the single source of truth for the document's type.

**Architecture:** Modifies the parsing service contract to return `tuple[list[str], SourceType]` instead of just `list[str]`. This cascades the document's original type directly into the chunking configuration, meaning the chunker doesn't have to duplicate the work of looking up the file path and inferring the extension.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pytest

## Global Constraints
- Do not introduce breaking changes to the `/parse` endpoint.
- Keep services stateless.
- Do not update the checklist until all tests pass.

---

### Task 1: Wire SourceType into Parser and Chunker

**Files:**
- Modify: `app/services/document_parser.py`
- Modify: `app/services/document_chunker.py`
- Modify: `app/api/v1/parse.py`
- Create/Modify: `tests/services/test_document_parser.py`
- Modify: `tests/services/test_document_chunker.py`

**Interfaces:**
- Consumes: `DocumentExtension` and `SourceType` from `app.core`.
- Produces: `parse_document` returning `tuple[list[str], SourceType]`.

- [ ] **Step 1: Write/Update the failing tests (RED)**

**In `tests/services/test_document_parser.py`:**
Create a direct unit test asserting `parse_document` returns the tuple:
```python
import pytest
from uuid import uuid4
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core import SourceType, DocumentExtension
from app.services.document_parser import parse_document


@pytest.mark.asyncio
async def test_parse_document_returns_text_and_source_type():
    doc_id = uuid4()
    with (
        patch("app.services.document_parser.find_document_path") as mock_find,
        patch("app.services.document_parser.get_parser") as mock_get_parser,
    ):
        mock_path = Path(f"/tmp/{doc_id}.pdf")
        mock_find.return_value = mock_path

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = ["Page 1 text"]
        mock_get_parser.return_value = mock_parser_instance

        pages, source_type = await parse_document(doc_id, Path("/tmp"))

        assert pages == ["Page 1 text"]
        assert source_type == SourceType.PDF
```

**In `tests/services/test_document_chunker.py`:**
Update the `mock_parse.return_value` to be a tuple `(["This is page one."], SourceType.TXT)` instead of a bare list. 
**Crucially, ensure there is NO patch for `find_document_path` anywhere in this file, as `chunk_document` should absolutely not be doing duplicate file lookups anymore.**

```python
with patch(
    "app.services.document_chunker.parse_document", new_callable=AsyncMock
) as mock_parse:
    mock_parse.return_value = (["This is page one."], SourceType.TXT)
```

**In `tests/api/test_parse.py` (if it exists) or any other tests masking `parse_document`:**
Review and update any mocks to return the tuple structure.

- [ ] **Step 2: Run test to verify it fails (RED)**

Run: `uv run pytest tests/services/test_document_parser.py tests/services/test_document_chunker.py -v`
Expected: FAIL due to missing tuple unpacking or incorrect return types.

- [ ] **Step 3: Write minimal implementation (GREEN)**

**In `app/services/document_parser.py`:**
Modify the signature and return statement of `parse_document`:
```python
from ..core import (
    DocumentExtension,
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
    SourceType,
)


async def parse_document(
    document_id: UUID, upload_dir: Path
) -> tuple[list[str], SourceType]:
    # ... existing lookup ...
    try:
        extension = DocumentExtension(path.suffix.lower())
    except ValueError:
        raise UnsupportedDocumentTypeError(path.suffix) from None

    parser = get_parser(extension)
    pages = await to_thread.run_sync(parser.parse, path)
    source_type = SourceType(extension.value.lstrip("."))

    return pages, source_type
```

**In `app/services/document_chunker.py`:**
Remove the hardcoded `SourceType.TXT` and unpack the tuple from `parse_document`:
```python
    pages, source_type = await parse_document(
        document_id=document_id,
        upload_dir=upload_dir,
    )
    
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_chars=settings.MIN_CHUNK_CHARS,
        source_type=source_type,  # Dynamically inferred!
    )
```

**In `app/api/v1/parse.py`:**
Update the API endpoint to unpack the tuple and retain the same external response contract:
```python
    pages, _ = await parse_document(document_id, settings.UPLOAD_DIR)
    return ParsedDocument(
        document_id=document_id,
        status=DocumentStatus.PARSING,
        pages=pages,
    )
```

- [ ] **Step 4: Run test to verify it passes (GREEN)**

Run: `uv run pytest tests/services/test_document_parser.py tests/services/test_document_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Run full verification (VERIFY)**

Run: 
```bash
uv run ruff check .
uv run mypy .
uv run pytest -v
```
Expected: All tests pass. 

- [ ] **Step 6: Update checklist and commit (REFACTOR/VERIFY)**

Check off the related boxes in `backend/checklist.md`:
- `Remove any duplicated compute between endpoints and services.`
- `Ensure the indexing pipeline is the single source of truth for ingestion.`

```bash
git add .
git commit -m "feat(ingestion): wire source_type inference and remove duplicated compute"
```

---

## Test Plan

- **New Tests**: `tests/services/test_document_parser.py::test_parse_document_returns_text_and_source_type` - directly validates the tuple return contract so we aren't just relying on integration tests.
- **Updated Tests**: `tests/services/test_document_chunker.py::test_chunk_document_propagates_source_type`. It must now mock the exact return signature (`tuple[list[str], SourceType]`) of `parse_document`. We explicitly removed any patching of `find_document_path` here.
- **Regression Tests**: All API endpoints (`/chunk`, `/index`, `/parse`) naturally exercise `chunk_document` and `parse_document`. Running the full test suite acts as an integration regression check. The reason these tests exist is to guarantee the wiring works end-to-end without mocking.

## Risks

1. **API Endpoint Drift:** Modifying `parse_document` breaks its signature. Any existing consumer that expects a `list[str]` will crash.
   - *Mitigation:* We explicitly update `app/api/v1/parse.py` and `app/services/document_chunker.py` to unpack the tuple. The integration tests (e.g., `tests/api/test_chunks.py`) will catch any unpacking errors immediately.
2. **Missing Endpoint Mock Updates:** If there were tests mocking `parse_document` for the parse endpoint, they would fail.
   - *Mitigation:* We verified that `test_parse.py` does not currently exist. If the host environment has it, the plan instructs it to review and update any mocks to return the tuple structure.
3. **Invalid `SourceType` mapping:** If an extension exists but has no matching `SourceType`, it could raise a `ValueError` during mapping.
   - *Mitigation:* The `SourceType` enum exactly mirrors `DocumentExtension` (`.pdf -> pdf`, `.txt -> txt`, `.md -> md`), ensuring the stripped extension string perfectly maps.

## Verification Checklist

- [ ] `uv run ruff check .` passes without errors.
- [ ] `uv run mypy .` passes without errors.
- [ ] Affected test suites (`tests/services/test_document_parser.py`, `tests/services/test_document_chunker.py`) pass.
- [ ] Full test suite (`uv run pytest -v`) passes.
- [ ] `backend/checklist.md` updated correctly.
- [ ] Git diff reviewed for any unintentional changes.
