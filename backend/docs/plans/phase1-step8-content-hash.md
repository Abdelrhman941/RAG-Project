# Phase 1: Step 8 Implementation Plan
> **Scope:** Cross-document duplicate prevention via content-hash lookup — fully implemented, not just stored.

**Goal:** Before embedding a chunk, compute `content_hash = SHA-256(content)`. Add `get_existing_hashes()` to `BaseVectorStore`. Implement it in `QdrantVectorStore`. In `index_document`, query existing hashes, filter them out, embed only new chunks, then upsert. Only new content reaches the embedding model and the vector store.

**Architecture:**
```
index_document()
  ├─ chunk_document()              ← Step 7 intra-doc dedup already applied
  ├─ compute content_hash per chunk
  ├─ vector_store.get_existing_hashes(collection, hashes)  ← NEW abstract method
  ├─ filter: keep only chunks whose hash is NOT already indexed
  ├─ embed_chunks(new_chunks_only) ← no wasted embedding calls
  └─ upsert(new_points_only)       ← no duplicate vectors in store
```

**Files touched:**
| File | Change |
|------|--------|
| `app/vectorstores/models.py` | Add `content_hash: str` to `PointPayload` |
| `app/schemas/chunk.py` | Add `content_hash: str` to `Chunk` |
| `app/vectorstores/base.py` | Add abstract `get_existing_hashes()` |
| `app/vectorstores/qdrant.py` | Implement `get_existing_hashes()` via payload scroll+filter |
| `app/services/document_chunker.py` | Compute `content_hash` when building each `Chunk` |
| `app/services/document_indexer.py` | Filter by hash before embed + upsert; propagate hash to payload |
| `tests/vectorstores/test_content_hash.py` | New: `PointPayload` hash field tests |
| `tests/services/test_document_indexer.py` | New: hash propagation + skip-if-existing tests |
| `tests/vectorstores/test_qdrant_get_hashes.py` | New: Qdrant adapter mock test for `get_existing_hashes` |

**Tech Stack:** Python 3.12, `hashlib` (stdlib), Qdrant payload scroll filter, pytest

---

## Global Constraints
- `BaseVectorStore` gains exactly **one** new method: `get_existing_hashes`.
- No vendor-specific (Qdrant) logic enters the service layer.
- No cache introduced.
- No semantic/embedding-based dedup introduced.
- Do not touch P2+ items.

---

## Task 1 — Extend models: `content_hash` in `PointPayload` and `Chunk`

### Step 1.1 — Write failing tests (RED)
Create `tests/vectorstores/test_content_hash.py`:
```python
import hashlib
import pytest
from pydantic import ValidationError
from uuid import uuid4
from app.core import SourceType
from app.vectorstores.models import PointPayload


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_point_payload_includes_content_hash() -> None:
    text = "Some chunk content."
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content=text,
        source_type=SourceType.TXT,
        start_char=0,
        end_char=len(text),
        content_hash=_sha256(text),
    )
    assert payload.content_hash == _sha256(text)


def test_point_payload_rejects_missing_content_hash() -> None:
    with pytest.raises(ValidationError):
        PointPayload(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            content="text",
            source_type=SourceType.TXT,
            start_char=0,
            end_char=4,
        )
```

### Step 1.2 — Verify RED
```bash
uv run pytest tests/vectorstores/test_content_hash.py -v
```
Expected: `ValidationError` or `TypeError` — `content_hash` field missing.

### Step 1.3 — Implement (GREEN)
In `app/vectorstores/models.py`, add after `end_char`:
```python
content_hash: str  # SHA-256 hex digest of chunk content
```
In `app/schemas/chunk.py`, add the same field to `Chunk`:
```python
content_hash: str  # SHA-256 hex digest of chunk content
```

### Step 1.4 — Verify GREEN
```bash
uv run pytest tests/vectorstores/test_content_hash.py -v
```

---

## Task 2 — Extend `BaseVectorStore` with `get_existing_hashes`

### Step 2.1 — Write failing test (RED)
Create `tests/vectorstores/test_qdrant_get_hashes.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.vectorstores.qdrant import QdrantVectorStore


@pytest.mark.asyncio
async def test_get_existing_hashes_returns_matched_hashes() -> None:
    store = QdrantVectorStore.__new__(QdrantVectorStore)

    # Fake a scroll response: two points with content_hash in payload
    fake_point_1 = MagicMock()
    fake_point_1.payload = {"content_hash": "aabbcc"}
    fake_point_2 = MagicMock()
    fake_point_2.payload = {"content_hash": "ddeeff"}

    store._client = AsyncMock()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.scroll = AsyncMock(return_value=([fake_point_1, fake_point_2], None))

    result = await store.get_existing_hashes(
        collection_name="test",
        hashes=frozenset({"aabbcc", "zzzzzz"}),
    )
    assert result == frozenset({"aabbcc"})


@pytest.mark.asyncio
async def test_get_existing_hashes_empty_when_collection_missing() -> None:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = AsyncMock()
    store._client.collection_exists = AsyncMock(return_value=False)

    result = await store.get_existing_hashes(
        collection_name="missing",
        hashes=frozenset({"aabbcc"}),
    )
    assert result == frozenset()
```

### Step 2.2 — Verify RED
```bash
uv run pytest tests/vectorstores/test_qdrant_get_hashes.py -v
```
Expected: `AttributeError` — method does not exist yet.

### Step 2.3 — Implement (GREEN)

**In `app/vectorstores/base.py`**, add abstract method:
```python
@abstractmethod
async def get_existing_hashes(
    self,
    collection_name: str,
    hashes: frozenset[str],
) -> frozenset[str]:
    """Return the subset of *hashes* that already exist in the collection.

    Used by the indexer to skip embedding and upserting chunks whose
    content has already been indexed.  Implementations must query only
    the `content_hash` payload field and must not return vendor types.
    """
```

**In `app/vectorstores/qdrant.py`**, add implementation:
```python
async def get_existing_hashes(
    self,
    collection_name: str,
    hashes: frozenset[str],
) -> frozenset[str]:
    """Query Qdrant for the subset of *hashes* already present.

    Uses a payload filter scroll so we only fetch matching points,
    keeping network overhead proportional to the number of new chunks.
    Returns an empty frozenset if the collection does not exist yet.
    """
    if not hashes:
        return frozenset()
    try:
        exists = await self._client.collection_exists(collection_name)
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc
    if not exists:
        return frozenset()

    try:
        points, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="content_hash",
                        match=qmodels.MatchAny(any=list(hashes)),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=len(hashes),
        )
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise VectorStoreUnavailableError(f"Qdrant hash lookup failed: {exc}") from exc

    found: set[str] = set()
    for point in points:
        if point.payload and "content_hash" in point.payload:
            found.add(str(point.payload["content_hash"]))
    return frozenset(found)
```

### Step 2.4 — Verify GREEN
```bash
uv run pytest tests/vectorstores/test_qdrant_get_hashes.py -v
```

---

## Task 3 — Compute hash in chunker; filter + embed only new in indexer

### Step 3.1 — Write failing tests (RED)
Create `tests/services/test_document_indexer.py`:
```python
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.core import ChunkingStrategy, DistanceMetric, SourceType
from app.schemas import Chunk
from app.services.document_indexer import _build_points, index_document


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content=text,
        start_char=0,
        end_char=len(text),
        char_count=len(text),
        source_type=SourceType.TXT,
        content_hash=_sha256(text),
    )


def test_build_points_propagates_content_hash() -> None:
    chunk = _make_chunk("Hello world.")
    points = _build_points([chunk], [[0.1] * 3])
    assert points[0].payload.content_hash == chunk.content_hash


@pytest.mark.asyncio
async def test_index_document_skips_already_indexed_chunks() -> None:
    """Chunks whose hash is already in the store must not be embedded or upserted."""
    existing_text = "Already indexed content."
    new_text = "Brand new content that is unique."

    doc_id = uuid4()
    mock_store = AsyncMock()
    mock_store.get_existing_hashes = AsyncMock(
        return_value=frozenset({_sha256(existing_text)})
    )
    mock_store.create_collection = AsyncMock()
    mock_store.delete_by_document = AsyncMock()
    mock_store.upsert = AsyncMock(return_value=1)

    mock_provider = MagicMock()
    mock_provider.embedding_dimension = 3
    mock_provider.model_name = "test-model"

    chunks = [_make_chunk(existing_text), _make_chunk(new_text)]

    with (
        patch(
            "app.services.document_indexer.chunk_document",
            new=AsyncMock(return_value=chunks),
        ),
        patch(
            "app.services.document_indexer.embed_chunks",
            new=AsyncMock(return_value=[[0.1] * 3]),
        ) as mock_embed,
    ):
        await index_document(
            document_id=doc_id,
            upload_dir=MagicMock(),
            provider=mock_provider,
            vector_store=mock_store,
            collection_name="test",
            distance=DistanceMetric.COSINE,
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=200,
            overlap=0,
        )

    # embed_chunks should only have been called with the 1 new chunk
    call_chunks = mock_embed.call_args[0][0]
    assert len(call_chunks) == 1
    assert call_chunks[0].content == new_text

    # upsert should only have been called with 1 point
    upsert_points = mock_store.upsert.call_args[1]["points"]
    assert len(upsert_points) == 1
    assert upsert_points[0].payload.content_hash == _sha256(new_text)
```

### Step 3.2 — Verify RED
```bash
uv run pytest tests/services/test_document_indexer.py -v
```

### Step 3.3 — Implement (GREEN)

**In `app/services/document_chunker.py`**, compute `content_hash` when building each chunk:
```python
import hashlib

content_hash = hashlib.sha256(span.content.encode()).hexdigest()
candidate = Chunk(
    ...,
    content_hash=content_hash,
)
```

**In `app/services/document_indexer.py`**, refactor `index_document`:
```python
async def index_document(...) -> IndexingResponse:
    chunks = await chunk_document(...)
    if not chunks:
        raise IndexingError(...)

    # --- Hash-based cross-document dedup ---
    candidate_hashes = frozenset(c.content_hash for c in chunks)
    existing_hashes = await vector_store.get_existing_hashes(
        collection_name, candidate_hashes
    )
    new_chunks = [c for c in chunks if c.content_hash not in existing_hashes]

    if new_chunks:
        vectors = await embed_chunks(new_chunks, provider)
        points = _build_points(new_chunks, vectors)
    else:
        points = []

    dimension = provider.embedding_dimension
    await vector_store.create_collection(...)
    await vector_store.delete_by_document(...)

    indexed = await vector_store.upsert(collection_name=collection_name, points=points)
    return IndexingResponse(
        ...,
        total_chunks=len(chunks),
        indexed_points=indexed,
    )
```

**In `_build_points`**, propagate `content_hash`:
```python
payload = PointPayload(
    ...,
    content_hash=chunk.content_hash,
)
```

### Step 3.4 — Verify GREEN
```bash
uv run pytest tests/services/test_document_indexer.py -v
```

---

## Task 4 — Full verification and checklist update

### Step 4.1 — Run full suite
```bash
uv run ruff check .
uv run mypy .
uv run pytest -v
```
Fix all regressions (any existing test constructing `Chunk` or `PointPayload` without `content_hash`).

### Step 4.2 — Update checklist
Only after all tests pass, mark:
```
[x] Prevent repeated or near-duplicate content from being indexed multiple times.
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| `content_hash` is required → existing tests break | Fix all call-sites in Step 4.1 |
| Qdrant `MatchAny` behaviour with many hashes | Limit scroll to `len(hashes)`; Qdrant handles up to thousands of match values |
| `delete_by_document` + hash-filter order | `delete_by_document` still runs to clear old chunks of the same document; hash filter adds cross-doc safety on top |
| All chunks already indexed → `new_chunks = []` | `embed_chunks([])` returns `[]`; `upsert([])` returns 0; `IndexingResponse.indexed_points = 0` is valid |

## Verification Checklist
- [ ] `ruff check .` passes.
- [ ] `mypy .` passes (pre-existing 2 errors excluded).
- [ ] `tests/vectorstores/test_content_hash.py` — 2 tests GREEN.
- [ ] `tests/vectorstores/test_qdrant_get_hashes.py` — 2 tests GREEN.
- [ ] `tests/services/test_document_indexer.py` — 2 tests GREEN.
- [ ] Full pytest suite passes.
- [ ] `checklist.md` item marked `[x]`.
