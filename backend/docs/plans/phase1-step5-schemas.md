# Phase 1, Step 5: `Chunk` Schema and `PointPayload` Updates

## Objective
Propagate the `source_type` down from the `ChunkSpan` through to the `Chunk` schema and eventually into the `PointPayload` that reaches the vector store, along with `start_char` and `end_char`.

## Scope
- Modify `app/schemas/chunk.py` to require `source_type`.
- Modify `app/vectorstores/models.py` to require `source_type`, `start_char`, and `end_char` in `PointPayload`.
- Create tests for both modifications.

## Files to Modified/Created
- `app/schemas/chunk.py` (modify)
- `app/vectorstores/models.py` (modify)
- `tests/schemas/__init__.py` (create)
- `tests/schemas/test_chunk_schema.py` (create)
- `tests/vectorstores/__init__.py` (create)
- `tests/vectorstores/test_point_payload.py` (create)

## Dependencies
- Consumes `SourceType` from `app.core`.
- Will temporarily break `/chunks` API and indexing pipeline downstream until Task 6 and Task 7 inject these fields during conversion. We will fix related breakage immediately if needed by updating mock instantiations in tests.

## Step-by-Step Implementation (TDD)

### Step 5.1: Write failing tests (RED)
Create `tests/schemas/test_chunk_schema.py`:
```python
import pytest
from uuid import uuid4
from app.schemas.chunk import Chunk
from app.core import SourceType

def test_chunk_accepts_source_type() -> None:
    chunk = Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="Hello world",
        start_char=0,
        end_char=11,
        char_count=11,
        source_type=SourceType.TXT,
    )
    assert chunk.source_type == SourceType.TXT

def test_chunk_rejects_missing_source_type() -> None:
    with pytest.raises(Exception):
        Chunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            page_number=1,
            content="hello",
            start_char=0,
            end_char=5,
            char_count=5,
        )
```

Create `tests/vectorstores/test_point_payload.py`:
```python
from uuid import uuid4
from app.vectorstores.models import PointPayload
from app.core import SourceType

def test_point_payload_carries_source_type_and_offsets() -> None:
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="hello",
        source_type=SourceType.PDF.value,
        start_char=0,
        end_char=5,
    )
    assert payload.source_type == "pdf"
    assert payload.start_char == 0
    assert payload.end_char == 5
```

### Step 5.2: Run failing tests
`uv run pytest tests/schemas/test_chunk_schema.py tests/vectorstores/test_point_payload.py -v`

### Step 5.3: Update `app/schemas/chunk.py` (GREEN)
- Import `SourceType` from `app.core`.
- Add `source_type: SourceType` as a required Pydantic field.

### Step 5.4: Update `app/vectorstores/models.py` (GREEN)
- Add the following to `PointPayload` (Pydantic model):
  - `source_type: str`
  - `start_char: int`
  - `end_char: int`

### Step 5.5: Run tests to verify
`uv run pytest tests/schemas/test_chunk_schema.py tests/vectorstores/test_point_payload.py -v`
Expected: Tests pass.

### Step 5.6: Fix Regressions
Run `uv run pytest -v`.
Address any breakage caused by updating the required fields in `Chunk` or `PointPayload` by adjusting the `chunk_document` service or test mocks before marking Step 5 complete.

### Step 5.7: Commit
Stage and commit changes.

## Acceptance Criteria
- `Chunk` requires `source_type`.
- `PointPayload` requires `source_type`, `start_char`, and `end_char`.
- Tests confirm validation rules.
