# Phase 1: Step 7 Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement **intra-document deduplication** only. Filter out highly similar repeated content within the same document (like boilerplates) during the chunking phase, using the `DEDUP_SIMILARITY_THRESHOLD` configuration. 

**Scope Warning:** 
- This does **not** resolve cross-corpus duplicate prevention.
- This does **not** resolve structural repeated headers/footers (which require parser-level layout awareness). Those items remain pending.

**Architecture:** We will create a `ChunkDeduplicator` utility in `app/chunkers/dedup.py` that uses `difflib.SequenceMatcher` to compare new chunks against already-accepted chunks in the same document. We will then wire this into `app/services/document_chunker.py` so that chunks are filtered *before* being returned. Filtering *before* embedding saves API/compute costs.

**Tech Stack:** Python 3.12, `difflib`, pytest

## Global Constraints
- Do not modify parsing or vector store schemas.
- Do not use embedding-based deduplication yet (this is string-based intra-document deduplication).
- Keep the $O(N^2)$ comparison efficient by fast-failing length checks before running `SequenceMatcher`.

---

### Task 1: Create the Deduplicator Utility

**Files:**
- Create: `app/chunkers/dedup.py`
- Create: `tests/chunkers/test_dedup.py`

- [ ] **Step 1: Write the failing tests (RED)**

Create `tests/chunkers/test_dedup.py`:
```python
from app.chunkers.dedup import is_duplicate
from app.core import SourceType
from app.schemas import Chunk
from uuid import uuid4


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
    )


def test_is_duplicate_exact_match():
    c1 = _make_chunk("This is a boilerplate disclaimer.")
    assert is_duplicate(c1, [c1], threshold=0.97)


def test_is_duplicate_highly_similar():
    c1 = _make_chunk("This is a boilerplate disclaimer. Page 1")
    c2 = _make_chunk("This is a boilerplate disclaimer. Page 2")
    assert is_duplicate(c2, [c1], threshold=0.85)


def test_is_not_duplicate_different_text():
    c1 = _make_chunk("This is completely unique content.")
    c2 = _make_chunk("This is a different paragraph.")
    assert not is_duplicate(c2, [c1], threshold=0.90)


def test_is_not_duplicate_empty_history():
    c1 = _make_chunk("First chunk")
    assert not is_duplicate(c1, [], threshold=0.90)
```

- [ ] **Step 2: Run test to verify it fails (RED)**
Run: `uv run pytest tests/chunkers/test_dedup.py -v` (Should fail with ImportError)

- [ ] **Step 3: Write minimal implementation (GREEN)**

Create `app/chunkers/dedup.py`:
```python
import difflib
from collections.abc import Sequence
from ..schemas import Chunk


def is_duplicate(new_chunk: Chunk, history: Sequence[Chunk], threshold: float) -> bool:
    """Check if new_chunk is highly similar to any chunk in history."""
    if not history:
        return False

    for past_chunk in history:
        # Fast fail: if length difference is too large, ratio can't mathematically meet threshold.
        len_new = len(new_chunk.content)
        len_past = len(past_chunk.content)
        if len_new == 0 and len_past == 0:
            continue

        max_possible_ratio = 2.0 * min(len_new, len_past) / (len_new + len_past)
        if max_possible_ratio < threshold:
            continue

        matcher = difflib.SequenceMatcher(None, new_chunk.content, past_chunk.content)
        # quick_ratio is a fast upper bound
        if matcher.quick_ratio() >= threshold:
            # actual ratio is more accurate but slower
            if matcher.ratio() >= threshold:
                return True

    return False
```

- [ ] **Step 4: Run test to verify it passes (GREEN)**
Run: `uv run pytest tests/chunkers/test_dedup.py -v`

---

### Task 2: Wire Deduplication into `document_chunker.py`

**Files:**
- Modify: `app/services/document_chunker.py`
- Modify: `tests/services/test_document_chunker.py`

- [ ] **Step 5: Write the failing tests (RED)**

Update `tests/services/test_document_chunker.py` to add deduplication and contiguous index tests:
```python
from unittest.mock import patch
from app.core import ChunkingStrategy
from pathlib import Path
from uuid import uuid4
from app.services.document_chunker import chunk_document
from app.core import SourceType


@pytest.mark.asyncio
async def test_chunk_document_filters_duplicates_and_maintains_contiguous_index() -> (
    None
):
    doc_id = uuid4()
    pages = [
        "Unique content on page 1.",
        "Boilerplate footer text.",
        "Unique content on page 2.",
        "Boilerplate footer text.",
    ]
    with patch("app.services.document_chunker.parse_document") as mock_parse:
        mock_parse.return_value = (pages, SourceType.TXT)

        chunks = await chunk_document(
            document_id=doc_id,
            upload_dir=Path("/tmp"),
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=10,
            overlap=0,
        )

        # 3 unique spans expected, duplicate footer dropped
        assert len(chunks) == 3
        contents = [c.content for c in chunks]
        assert contents.count("Boilerplate footer text.") == 1

        # Ensure contiguous indices: 0, 1, 2
        indices = [c.chunk_index for c in chunks]
        assert indices == [0, 1, 2]
```

- [ ] **Step 6: Run test to verify it fails (RED)**
Run: `uv run pytest tests/services/test_document_chunker.py -v` (Fails because it returns 4 chunks instead of 3, and indices are 0,1,2,3).

- [ ] **Step 7: Write minimal implementation (GREEN)**

In `app/services/document_chunker.py`, import `is_duplicate` and `get_settings`, then apply it:
```python
    from ..chunkers.dedup import is_duplicate
    
    # inside chunk_document loop:
    for span in chunker.chunk(page_text, config):
        new_chunk = Chunk(
            chunk_id=uuid4(),
            document_id=document_id,
            chunk_index=current_chunk_index,
            page_number=page_number,
            content=span.content,
            start_char=span.start_char,
            end_char=span.end_char,
            char_count=len(span.content),
            source_type=span.source_type,
        )
        
        if not is_duplicate(new_chunk, chunks, settings.DEDUP_SIMILARITY_THRESHOLD):
            chunks.append(new_chunk)
            current_chunk_index += 1
```

- [ ] **Step 8: Run test to verify it passes (GREEN)**
Run: `uv run pytest tests/services/test_document_chunker.py -v`

- [ ] **Step 9: Run full verification (VERIFY)**
Run `uv run ruff check .`, `uv run mypy .`, and `uv run pytest -v`.

- [ ] **Step 10: Update checklist and commit**
Check off `Deduplicate highly similar repeated content.` in `backend/checklist.md`.
*(Note: Do NOT check off 'repeated headers/footers' or 'prevent repeated content from being indexed multiple times', as those require parser structural layout and vector-level cross-document features, respectively).*
```bash
git add .
git commit -m "feat(chunking): implement string-based intra-document deduplication"
```

## Risks
- **Performance:** $O(N^2)$ string comparisons per document. Mitigated by `max_possible_ratio` fast-math check and `matcher.quick_ratio()`. For typical document sizes (50-500 chunks), this operates in milliseconds.
- **Index Misalignment:** If chunks are dropped, `chunk_index` might skip numbers if not handled correctly. We increment `current_chunk_index` *only* when a chunk is actually appended to ensure a contiguous index. This is explicitly verified by the new test.
