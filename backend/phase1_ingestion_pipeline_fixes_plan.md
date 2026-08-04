# Phase 1 — Ingestion Pipeline Fixes: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the `/index` ingestion pipeline so that it normalizes text, filters degenerate chunks, enriches chunk metadata, and prevents near-duplicate content from being stored in the vector database.

**Architecture:** All changes are confined to the service and chunker layers. The API layer (`/index`, `/embed`) receives only the minimal wiring needed to thread new settings values through. `document_chunker.py` becomes the single point of authority for text normalization and chunk filtering. `document_indexer.py` gains a deduplication gate before the upsert step. `ChunkingConfig` and `Chunk` schema absorb only the new fields required by the spec.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, LangChain text splitters (tiktoken), pypdf, anyio, pytest, pytest-asyncio.

## Global Constraints

- `/index` (`POST /api/v1/documents/{id}/index`) is the production ingestion path; its **response schema must not change**.
- `/embed` (`POST /api/v1/documents/{id}/embed`) is a read-only inspection endpoint; it must continue to work but is not the primary pipeline.
- Services must remain stateless — no in-memory caches, no cross-request state.
- Do **not** introduce caching unless explicitly required by a task below.
- Follow the existing module/layer structure: parsers → chunkers → services → API.
- All new config values go into `app/core/config.py` (`Settings`) and are read from environment variables via `pydantic-settings`.
- All new exceptions extend the existing hierarchy in `app/core/exceptions.py`.
- Coding style: dataclasses with `frozen=True, slots=True` for value objects; Pydantic v2 `BaseModel` with `ConfigDict(frozen=True)` for schemas; `@abstractmethod` for base classes.
- Test runner: `pytest` from the `backend/` directory; async tests use `pytest-asyncio`.
- Re-indexing policy: any change to chunking logic must be noted as requiring a re-index; existing collections indexed with old logic become stale. This plan does not automate re-indexing.

---

## Files to Be Modified or Created

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `app/chunkers/normalizer.py` | Pure function: normalize line endings, whitespace, excessive blank lines |
| Modify | `app/chunkers/__init__.py` | Re-export `normalize_text` |
| Modify | `app/chunkers/models.py` | Add `min_chunk_tokens` to `ChunkingConfig`; add `source_type` to `ChunkSpan` |
| Modify | `app/chunkers/token.py` | Apply normalization before splitting; add small-chunk merge pass |
| Modify | `app/core/config.py` | Add `MIN_CHUNK_TOKENS`, `DEDUP_SIMILARITY_THRESHOLD` settings |
| Modify | `app/core/enums/document.py` | Add `SourceType` enum |
| Modify | `app/core/enums/__init__.py` | Re-export `SourceType` |
| Modify | `app/core/__init__.py` | Re-export `SourceType` |
| Modify | `app/schemas/chunk.py` | Add `source_type: SourceType` field to `Chunk` |
| Modify | `app/vectorstores/models.py` | Add `source_type: str`, `start_char: int`, `end_char: int` to `PointPayload` |
| Modify | `app/services/document_chunker.py` | Accept `min_chunk_tokens`; infer and pass `source_type`; add empty-span guard |
| Modify | `app/services/document_indexer.py` | Add `_deduplicate_chunks`; accept `min_chunk_tokens` + `dedup_threshold`; enrich `PointPayload` |
| Modify | `app/services/document_embedder.py` | Accept `min_chunk_tokens`; thread to `chunk_document` |
| Modify | `app/api/v1/index.py` | Pass `min_chunk_tokens` + `dedup_threshold` from settings |
| Modify | `app/api/v1/embed.py` | Pass `min_chunk_tokens` from settings |
| Create | `tests/__init__.py` | Package marker |
| Create | `tests/chunkers/__init__.py` | Package marker |
| Create | `tests/chunkers/test_normalizer.py` | Unit tests for the normalizer |
| Create | `tests/chunkers/test_models.py` | Unit tests for updated `ChunkingConfig`/`ChunkSpan` |
| Create | `tests/chunkers/test_token_chunker.py` | Unit tests for normalization + merge pass |
| Create | `tests/core/__init__.py` | Package marker |
| Create | `tests/core/test_config.py` | Unit tests for new settings fields |
| Create | `tests/schemas/__init__.py` | Package marker |
| Create | `tests/schemas/test_chunk_schema.py` | Unit tests for `Chunk` with `source_type` |
| Create | `tests/vectorstores/__init__.py` | Package marker |
| Create | `tests/vectorstores/test_point_payload.py` | Unit tests for `PointPayload` with new fields |
| Create | `tests/services/__init__.py` | Package marker |
| Create | `tests/services/test_document_chunker.py` | Unit tests for `chunk_document` with mocked parser |
| Create | `tests/services/test_document_indexer.py` | Unit tests for `_deduplicate_chunks` and `_build_points` |
| Create | `tests/api/__init__.py` | Package marker |
| Create | `tests/api/test_index_endpoint_smoke.py` | Smoke test: route is registered and wired |

> **Note to implementer:** there are currently **no** `tests/` directories under `backend/`. Create `tests/` with `__init__.py` files as needed. Mirror the `app/` package structure under `tests/`.

---

## Task 1 — Text Normalization Module

**Files:**
- Create: `app/chunkers/normalizer.py`
- Create: `tests/__init__.py`, `tests/chunkers/__init__.py`, `tests/chunkers/test_normalizer.py`
- Modify: `app/chunkers/__init__.py`

**Interfaces:**
- Produces: `normalize_text(text: str) -> str` — a pure, stateless function used by `RecursiveChunker.chunk()` in Task 4.

---

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/chunkers/test_normalizer.py
import pytest
from app.chunkers.normalizer import normalize_text


def test_crlf_normalized_to_lf():
    assert normalize_text("a\r\nb") == "a\nb"


def test_cr_normalized_to_lf():
    assert normalize_text("a\rb") == "a\nb"


def test_repeated_whitespace_collapsed():
    assert normalize_text("hello   world") == "hello world"


def test_tabs_normalized():
    assert normalize_text("hello\tworld") == "hello world"


def test_repeated_blank_lines_collapsed():
    result = normalize_text("a\n\n\n\nb")
    assert result == "a\n\nb"


def test_leading_trailing_whitespace_stripped():
    assert normalize_text("  hello  ") == "hello"


def test_empty_string_returns_empty():
    assert normalize_text("") == ""


def test_only_whitespace_returns_empty():
    assert normalize_text("   \n\t  ") == ""


def test_already_normalized_is_unchanged():
    text = "First paragraph.\n\nSecond paragraph."
    assert normalize_text(text) == text
```

- [ ] **Step 1.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/chunkers/test_normalizer.py -v
```

Expected: `ImportError` — `app.chunkers.normalizer` does not exist yet.

- [ ] **Step 1.3: Implement `normalize_text`**

```python
# app/chunkers/normalizer.py
"""Pure text normalization for the ingestion pipeline.

Called once per page text, before any chunking strategy is applied.
All transformations are deterministic and stateless.
"""

import re

# Three or more consecutive newlines → two (one blank line)
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
# Runs of horizontal whitespace (spaces + tabs) inside a line → single space
_INLINE_WHITESPACE_RE = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Normalize *text* for chunking.

    Transformations applied in order:
    1. Normalize line endings to ``\\n``.
    2. Collapse repeated horizontal whitespace within lines to a single space.
    3. Collapse three or more consecutive newlines to two (one blank line).
    4. Strip leading/trailing whitespace from the result.
    """
    if not text:
        return text

    # 1. Line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. Inline whitespace (tabs, multiple spaces) – do not touch newlines
    text = _INLINE_WHITESPACE_RE.sub(" ", text)
    # 3. Collapse excessive blank lines
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    # 4. Strip
    return text.strip()
```

- [ ] **Step 1.4: Export from `app/chunkers/__init__.py`**

  Add to `app/chunkers/__init__.py`:
  ```python
  from .normalizer import normalize_text
  ```
  Add `"normalize_text"` to `__all__`.

- [ ] **Step 1.5: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/chunkers/test_normalizer.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add app/chunkers/normalizer.py app/chunkers/__init__.py \
        tests/__init__.py tests/chunkers/__init__.py \
        tests/chunkers/test_normalizer.py
git commit -m "feat(chunker): add pure text normalizer module"
```

---

## Task 2 — Config: `MIN_CHUNK_TOKENS` and `DEDUP_SIMILARITY_THRESHOLD`

**Files:**
- Modify: `app/core/config.py`
- Create: `tests/core/__init__.py`, `tests/core/test_config.py`

**Interfaces:**
- Produces: `settings.MIN_CHUNK_TOKENS: int` (default `10`) and `settings.DEDUP_SIMILARITY_THRESHOLD: float` (default `0.97`) consumed by Tasks 3 and 7.

---

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/core/test_config.py
from app.core.config import Settings


def test_min_chunk_tokens_has_sensible_default():
    s = Settings()
    assert s.MIN_CHUNK_TOKENS == 10
    assert s.MIN_CHUNK_TOKENS > 0


def test_dedup_similarity_threshold_default():
    s = Settings()
    assert s.DEDUP_SIMILARITY_THRESHOLD == 0.97
    assert 0.0 < s.DEDUP_SIMILARITY_THRESHOLD <= 1.0
```

- [ ] **Step 2.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/core/test_config.py -v
```

Expected: `AttributeError` — fields do not exist yet.

- [ ] **Step 2.3: Add fields to `Settings`**

  In `app/core/config.py`, inside the `Settings` class, add after the `# ------------ Chunking ------------` block:
  ```python
  MIN_CHUNK_TOKENS: int = Field(default=10, gt=0)
  DEDUP_SIMILARITY_THRESHOLD: float = Field(default=0.97, ge=0.0, le=1.0)
  ```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/core/test_config.py -v
```

Expected: Both tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add app/core/config.py tests/core/__init__.py tests/core/test_config.py
git commit -m "feat(config): add MIN_CHUNK_TOKENS and DEDUP_SIMILARITY_THRESHOLD"
```

---

## Task 3 — `SourceType` Enum + Update `ChunkingConfig` and `ChunkSpan`

**Files:**
- Modify: `app/core/enums/document.py` — add `SourceType` enum
- Modify: `app/core/enums/__init__.py` — re-export `SourceType`
- Modify: `app/core/__init__.py` — re-export `SourceType`
- Modify: `app/chunkers/models.py` — add `min_chunk_tokens` + `source_type`
- Create: `tests/chunkers/test_models.py`

**Interfaces:**
- Consumes: `settings.MIN_CHUNK_TOKENS` (Task 2)
- Produces:
  - `SourceType` enum with members `PDF = "pdf"`, `TXT = "txt"`, `MD = "md"`
  - `ChunkingConfig.min_chunk_tokens: int`
  - `ChunkingConfig.source_type: SourceType`
  - `ChunkSpan.source_type: SourceType`

---

- [ ] **Step 3.1: Add `SourceType` enum to `app/core/enums/document.py`**

  Append to the existing enums file:
  ```python
  class SourceType(str, Enum):
      """The origin file type of a parsed document."""
      PDF = "pdf"
      TXT = "txt"
      MD = "md"
  ```

  Re-export from `app/core/enums/__init__.py` and `app/core/__init__.py`, following the exact same pattern already used for `DocumentStatus`, `ChunkingStrategy`, etc.

- [ ] **Step 3.2: Write the failing tests**

```python
# tests/chunkers/test_models.py
import pytest
from app.chunkers.models import ChunkingConfig, ChunkSpan
from app.core import ChunkingStrategy, SourceType


def test_chunking_config_accepts_min_chunk_tokens():
    cfg = ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        chunk_size=512,
        overlap=50,
        min_chunk_tokens=10,
        source_type=SourceType.PDF,
    )
    assert cfg.min_chunk_tokens == 10
    assert cfg.source_type == SourceType.PDF


def test_chunking_config_min_chunk_tokens_must_be_positive():
    with pytest.raises(Exception):
        ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=512,
            overlap=50,
            min_chunk_tokens=0,
            source_type=SourceType.TXT,
        )


def test_chunk_span_carries_source_type():
    span = ChunkSpan(
        content="hello", start_char=0, end_char=5, source_type=SourceType.MD
    )
    assert span.source_type == SourceType.MD
```

- [ ] **Step 3.3: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/chunkers/test_models.py -v
```

Expected: `TypeError` — unexpected keyword argument.

- [ ] **Step 3.4: Update `ChunkingConfig` and `ChunkSpan` in `app/chunkers/models.py`**

```python
from dataclasses import dataclass

from ..core import ChunkingStrategy, InvalidChunkingParametersError, SourceType


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """A chunk and its character offsets within the original text."""

    content: str
    start_char: int
    end_char: int
    source_type: SourceType


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration shared by all chunking strategies."""

    strategy: ChunkingStrategy
    chunk_size: int
    overlap: int
    min_chunk_tokens: int
    source_type: SourceType

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise InvalidChunkingParametersError("chunk_size must be greater than 0.")

        if self.overlap < 0:
            raise InvalidChunkingParametersError(
                "overlap must be greater than or equal to 0."
            )

        if self.overlap >= self.chunk_size:
            raise InvalidChunkingParametersError(
                "overlap must be smaller than chunk_size."
            )

        if self.min_chunk_tokens <= 0:
            raise InvalidChunkingParametersError(
                "min_chunk_tokens must be greater than 0."
            )
```

- [ ] **Step 3.5: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/chunkers/test_models.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 3.6: Run full test suite to check regressions**

```bash
cd backend
python -m pytest -v
```

  Fix any breakage caused by the new required fields before proceeding. The `/chunks` endpoint constructs `ChunkingConfig` indirectly through `chunk_document` — that will be fixed in Task 6.

- [ ] **Step 3.7: Commit**

```bash
git add app/core/enums/document.py app/core/enums/__init__.py app/core/__init__.py \
        app/chunkers/models.py \
        tests/chunkers/test_models.py
git commit -m "feat(chunker): add SourceType enum and min_chunk_tokens to ChunkingConfig"
```

---

## Task 4 — `RecursiveChunker`: Apply Normalization + Merge Small Chunks

**Files:**
- Modify: `app/chunkers/token.py`
- Create: `tests/chunkers/test_token_chunker.py`

**Interfaces:**
- Consumes: `normalize_text` (Task 1), `ChunkingConfig.min_chunk_tokens` and `ChunkingConfig.source_type` (Task 3).
- Produces: `RecursiveChunker.chunk(text, config) -> list[ChunkSpan]` where:
  - Text is normalized before splitting.
  - Each span carries `source_type` from `config`.
  - Chunks whose whitespace-split word count is below `config.min_chunk_tokens` are merged left (appended to the preceding chunk) or right (prepended to the next chunk). Any chunk still below threshold after both passes is dropped.

> **Merge pass note:** work on the raw string list returned by the splitter before building `ChunkSpan` objects. Merge by concatenating content with a single `"\n"`. Recompute `start_char`/`end_char` by searching the post-normalized `text` using `text.find(chunk_text, search_start)`, exactly as the original code does.

---

- [ ] **Step 4.1: Write the failing tests**

```python
# tests/chunkers/test_token_chunker.py
import pytest
from app.chunkers.token import RecursiveChunker
from app.chunkers.models import ChunkingConfig
from app.core import ChunkingStrategy, SourceType

PDF = SourceType.PDF


def _cfg(chunk_size=100, overlap=10, min_chunk_tokens=5):
    return ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
        source_type=PDF,
    )


def test_empty_text_returns_no_spans():
    chunker = RecursiveChunker()
    assert chunker.chunk("", _cfg()) == []


def test_whitespace_only_returns_no_spans():
    chunker = RecursiveChunker()
    assert chunker.chunk("   \n\t  ", _cfg()) == []


def test_normal_text_returns_spans_with_correct_source_type():
    chunker = RecursiveChunker()
    text = "word " * 200
    spans = chunker.chunk(text, _cfg(chunk_size=50, overlap=5, min_chunk_tokens=3))
    assert all(s.source_type == PDF for s in spans)


def test_tiny_chunk_is_merged_into_neighbor():
    chunker = RecursiveChunker()
    long_part = "word " * 60
    short_tail = "end"
    text = long_part + short_tail
    spans = chunker.chunk(text, _cfg(chunk_size=200, overlap=0, min_chunk_tokens=10))
    contents = [s.content for s in spans]
    assert not any(c.strip() == "end" for c in contents)


def test_span_offsets_are_within_original_text_bounds():
    chunker = RecursiveChunker()
    text = "Hello world. " * 40
    cfg = _cfg(chunk_size=50, overlap=5, min_chunk_tokens=3)
    spans = chunker.chunk(text, cfg)
    for span in spans:
        assert span.start_char >= 0
        assert span.end_char <= len(text)
        assert span.start_char < span.end_char


def test_crlf_in_input_is_normalized():
    chunker = RecursiveChunker()
    text = "Line one.\r\nLine two.\r\nLine three."
    spans = chunker.chunk(text, _cfg())
    combined = " ".join(s.content for s in spans)
    assert "\r" not in combined
```

- [ ] **Step 4.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/chunkers/test_token_chunker.py -v
```

Expected: `TypeError` — `ChunkSpan` constructor rejects old call site missing `source_type`.

- [ ] **Step 4.3: Implement normalization + merge pass in `RecursiveChunker`**

```python
# app/chunkers/token.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker
from .models import ChunkingConfig, ChunkSpan
from .normalizer import normalize_text


class RecursiveChunker(BaseChunker):
    """Token-aware recursive chunker with text normalization and small-chunk merging.

    Uses LangChain's RecursiveCharacterTextSplitter configured with a
    tokenizer-aware length function. The splitter tries to preserve
    paragraphs, lines, and sentences before falling back to characters.

    Before splitting, the input text is normalized (line endings, whitespace,
    blank lines). After splitting, chunks below ``config.min_chunk_tokens``
    are merged into their neighbor to prevent micro-chunks from reaching the
    vector store.
    """

    def chunk(
        self,
        text: str,
        config: ChunkingConfig,
    ) -> list[ChunkSpan]:
        text = normalize_text(text)
        if not text:
            return []

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
        )
        raw_chunks: list[str] = splitter.split_text(text)
        merged = self._merge_small_chunks(raw_chunks, config.min_chunk_tokens)

        spans: list[ChunkSpan] = []
        search_start = 0

        for chunk_text in merged:
            start = text.find(chunk_text, search_start)
            # fallback for whitespace normalization differences
            if start == -1:
                start = text.find(chunk_text)
            if start == -1:
                start = search_start
            end = start + len(chunk_text)
            spans.append(
                ChunkSpan(
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    source_type=config.source_type,
                )
            )
            search_start = end

        return spans

    @staticmethod
    def _merge_small_chunks(chunks: list[str], min_tokens: int) -> list[str]:
        """Merge chunks whose whitespace-split word count is below *min_tokens*.

        Pass 1 (left merge): scan left to right. When a chunk is too small,
        append it to the preceding chunk. If there is no preceding chunk,
        defer it.
        Pass 2 (right merge): if the deferred leading chunk is still small,
        prepend it to the next chunk.
        Drop any chunk that remains below threshold after both passes.
        """
        if not chunks:
            return chunks

        result: list[str] = []
        for chunk in chunks:
            token_count = len(chunk.split())
            if token_count >= min_tokens:
                result.append(chunk)
            elif result:
                result[-1] = result[-1] + "\n" + chunk
            else:
                result.append(chunk)  # no predecessor — defer for pass 2

        # Pass 2: forward-merge any remaining leading small chunk
        if len(result) > 1 and len(result[0].split()) < min_tokens:
            result[1] = result[0] + "\n" + result[1]
            result = result[1:]

        # Drop surviving micro-chunks (document is entirely tiny)
        return [c for c in result if len(c.split()) >= min_tokens]
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/chunkers/test_token_chunker.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 4.5: Run full test suite**

```bash
cd backend
python -m pytest -v
```

- [ ] **Step 4.6: Commit**

```bash
git add app/chunkers/token.py tests/chunkers/test_token_chunker.py
git commit -m "feat(chunker): normalize text and merge small chunks in RecursiveChunker"
```

---

## Task 5 — `Chunk` Schema: Add `source_type`; `PointPayload`: Add `source_type`, `start_char`, `end_char`

**Files:**
- Modify: `app/schemas/chunk.py`
- Modify: `app/vectorstores/models.py`
- Create: `tests/schemas/__init__.py`, `tests/schemas/test_chunk_schema.py`
- Create: `tests/vectorstores/__init__.py`, `tests/vectorstores/test_point_payload.py`

**Interfaces:**
- Consumes: `SourceType` (Task 3), `ChunkSpan.source_type` (Task 3).
- Produces:
  - `Chunk.source_type: SourceType`
  - `PointPayload.source_type: str` (stored as raw string to stay provider-agnostic)
  - `PointPayload.start_char: int`
  - `PointPayload.end_char: int`

---

- [ ] **Step 5.1: Write the failing tests**

```python
# tests/schemas/test_chunk_schema.py
import pytest
from uuid import uuid4
from app.schemas.chunk import Chunk
from app.core import SourceType


def _chunk(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return Chunk(**defaults)


def test_chunk_accepts_source_type():
    chunk = _chunk()
    assert chunk.source_type == SourceType.TXT


def test_chunk_rejects_missing_source_type():
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
            # source_type intentionally omitted
        )
```

```python
# tests/vectorstores/test_point_payload.py
from uuid import uuid4
from app.vectorstores.models import PointPayload
from app.core import SourceType


def test_point_payload_carries_source_type_and_offsets():
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

- [ ] **Step 5.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/schemas/test_chunk_schema.py tests/vectorstores/test_point_payload.py -v
```

Expected: `ValidationError` / `TypeError` — fields do not exist yet.

- [ ] **Step 5.3: Update `Chunk` schema in `app/schemas/chunk.py`**

  Add `source_type: SourceType` as a required field (no default). Update the import:
  ```python
  from ..core import ChunkingStrategy, DocumentStatus, SourceType
  ```

- [ ] **Step 5.4: Update `PointPayload` in `app/vectorstores/models.py`**

  Add three new required fields after `content`:
  ```python
  source_type: str                           # e.g. "pdf", "txt", "md"
  start_char: Annotated[int, Field(ge=0)]
  end_char: Annotated[int, Field(ge=0)]
  ```

- [ ] **Step 5.5: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/schemas/test_chunk_schema.py tests/vectorstores/test_point_payload.py -v
```

- [ ] **Step 5.6: Run full test suite**

```bash
cd backend
python -m pytest -v
```

  Fix any regressions caused by the new required fields (any code constructing `PointPayload` or `Chunk` must be updated).

- [ ] **Step 5.7: Commit**

```bash
git add app/schemas/chunk.py app/vectorstores/models.py \
        tests/schemas/__init__.py tests/schemas/test_chunk_schema.py \
        tests/vectorstores/__init__.py tests/vectorstores/test_point_payload.py
git commit -m "feat(schema): add source_type to Chunk; add source_type and offsets to PointPayload"
```

---

## Task 6 — `document_chunker.py`: Wire `source_type` + Empty-Chunk Guard

**Files:**
- Modify: `app/services/document_chunker.py`
- Create: `tests/services/__init__.py`, `tests/services/test_document_chunker.py`

**Interfaces:**
- Consumes:
  - `parse_document(document_id, upload_dir) -> list[str]` (unchanged)
  - `find_document_path(document_id, upload_dir) -> Path` (already in `document_parser.py`)
  - `ChunkingConfig(strategy, chunk_size, overlap, min_chunk_tokens, source_type)` (Task 3)
  - `SourceType`, `DocumentExtension` (existing enum in `app/core`)
- Produces: `chunk_document(..., min_chunk_tokens: int) -> list[Chunk]`
  - `Chunk.source_type` is inferred from the document's file extension.
  - Empty or whitespace-only spans are dropped before returning.

> **Extension → SourceType mapping** to add to the module:
> `DocumentExtension.PDF → SourceType.PDF`, `DocumentExtension.TXT → SourceType.TXT`, `DocumentExtension.MD → SourceType.MD`

---

- [ ] **Step 6.1: Write the failing tests**

```python
# tests/services/test_document_chunker.py
import pytest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch, AsyncMock

from app.services.document_chunker import chunk_document
from app.core import ChunkingStrategy, SourceType


@pytest.mark.asyncio
async def test_chunk_document_skips_empty_pages(tmp_path):
    doc_id = uuid4()
    pages = ["", "   ", "Hello world. " * 20, ""]

    with patch(
        "app.services.document_chunker.parse_document",
        new=AsyncMock(return_value=pages),
    ), patch(
        "app.services.document_chunker.find_document_path",
        return_value=tmp_path / f"{doc_id}.txt",
    ):
        chunks = await chunk_document(
            document_id=doc_id,
            upload_dir=tmp_path,
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=200,
            overlap=20,
            min_chunk_tokens=5,
        )

    assert len(chunks) > 0
    assert all(c.source_type == SourceType.TXT for c in chunks)
    assert all(c.content.strip() for c in chunks)


@pytest.mark.asyncio
async def test_chunk_document_all_empty_pages_returns_empty(tmp_path):
    doc_id = uuid4()
    pages = ["", "   "]

    with patch(
        "app.services.document_chunker.parse_document",
        new=AsyncMock(return_value=pages),
    ), patch(
        "app.services.document_chunker.find_document_path",
        return_value=tmp_path / f"{doc_id}.txt",
    ):
        chunks = await chunk_document(
            document_id=doc_id,
            upload_dir=tmp_path,
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=200,
            overlap=20,
            min_chunk_tokens=5,
        )

    assert chunks == []
```

- [ ] **Step 6.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/services/test_document_chunker.py -v
```

Expected: `TypeError` — `chunk_document` does not accept `min_chunk_tokens` yet.

- [ ] **Step 6.3: Rewrite `app/services/document_chunker.py`**

```python
from pathlib import Path
from uuid import UUID, uuid4

from ..chunkers import ChunkingConfig, get_chunker
from ..core import ChunkingStrategy, DocumentExtension, SourceType
from ..schemas import Chunk
from .document_parser import find_document_path, parse_document


_EXT_TO_SOURCE_TYPE: dict[DocumentExtension, SourceType] = {
    DocumentExtension.PDF: SourceType.PDF,
    DocumentExtension.TXT: SourceType.TXT,
    DocumentExtension.MD: SourceType.MD,
}


async def chunk_document(
    document_id: UUID,
    upload_dir: Path,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    """Parse a document then split it into ordered, filtered chunks.

    Empty pages are skipped before chunking. The ``source_type`` for
    every chunk is inferred from the document's file extension.
    """
    path = find_document_path(document_id, upload_dir)
    extension = DocumentExtension(path.suffix.lower())
    source_type = _EXT_TO_SOURCE_TYPE[extension]

    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
        source_type=source_type,
    )
    pages = await parse_document(
        document_id=document_id,
        upload_dir=upload_dir,
    )
    chunker = get_chunker(config.strategy)
    chunks: list[Chunk] = []
    current_chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        for span in chunker.chunk(page_text, config):
            if not span.content.strip():
                continue  # defensive guard: drop empty spans

            chunks.append(
                Chunk(
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
            )
            current_chunk_index += 1

    return chunks
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/services/test_document_chunker.py -v
```

- [ ] **Step 6.5: Run full test suite**

```bash
cd backend
python -m pytest -v
```

  Fix any regressions (call sites of `chunk_document` now need `min_chunk_tokens`).

- [ ] **Step 6.6: Commit**

```bash
git add app/services/document_chunker.py \
        tests/services/__init__.py tests/services/test_document_chunker.py
git commit -m "feat(service): wire source_type and min_chunk_tokens into chunk_document"
```

---

## Task 7 — `document_indexer.py`: Near-Duplicate Filter + Enrich `PointPayload`

**Files:**
- Modify: `app/services/document_indexer.py`
- Create: `tests/services/test_document_indexer.py`

**Interfaces:**
- Consumes:
  - `chunk_document(..., min_chunk_tokens)` (Task 6)
  - `Chunk.source_type`, `Chunk.start_char`, `Chunk.end_char` (Task 5)
  - `PointPayload` with new fields (Task 5)
- Produces: `index_document(...)` — same `IndexingResponse` return type; now accepts two new parameters: `min_chunk_tokens: int` and `dedup_threshold: float`.

> **Deduplication strategy:** O(n²) trigram Jaccard over normalized (lowercased, collapsed-whitespace) content strings. No external dependency needed. The first occurrence of each near-duplicate group is kept.
>
> Trigram Jaccard formula: `|A ∩ B| / |A ∪ B|` where A and B are sets of 3-character substrings.
>
> If `jaccard(normalized_a, normalized_b) >= threshold`, the candidate is dropped.

---

- [ ] **Step 7.1: Write the failing tests**

```python
# tests/services/test_document_indexer.py
import pytest
from uuid import uuid4
from app.services.document_indexer import _deduplicate_chunks, _build_points
from app.schemas.chunk import Chunk
from app.core import SourceType
from app.vectorstores.models import PointPayload


def _make_chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=index,
        page_number=1,
        content=content,
        start_char=0,
        end_char=len(content),
        char_count=len(content),
        source_type=SourceType.PDF,
    )


def test_deduplicate_removes_near_identical_chunks():
    chunks = [
        _make_chunk("The quick brown fox jumps over the lazy dog", 0),
        _make_chunk("The quick brown fox jumps over the lazy dog", 1),  # exact dupe
    ]
    result = _deduplicate_chunks(chunks, threshold=0.97)
    assert len(result) == 1


def test_deduplicate_keeps_distinct_chunks():
    chunks = [
        _make_chunk("Machine learning is a subset of artificial intelligence.", 0),
        _make_chunk("The capital of France is Paris.", 1),
    ]
    result = _deduplicate_chunks(chunks, threshold=0.97)
    assert len(result) == 2


def test_deduplicate_empty_list():
    assert _deduplicate_chunks([], threshold=0.97) == []


def test_build_points_populates_source_type_and_offsets():
    chunk = _make_chunk("hello world", 0)
    points = _build_points([chunk], [[0.1] * 128])
    payload: PointPayload = points[0].payload
    assert payload.source_type == SourceType.PDF.value
    assert payload.start_char == 0
    assert payload.end_char == 11
```

- [ ] **Step 7.2: Run to verify tests fail**

```bash
cd backend
python -m pytest tests/services/test_document_indexer.py -v
```

Expected: `ImportError` — `_deduplicate_chunks` is not yet defined.

- [ ] **Step 7.3: Rewrite `app/services/document_indexer.py`**

```python
import re
from pathlib import Path
from uuid import UUID, uuid4

from ..core import (
    ChunkingStrategy,
    DistanceMetric,
    DocumentStatus,
    IndexingError,
)
from ..embedders import BaseEmbeddingProvider
from ..schemas import Chunk, IndexingResponse
from ..vectorstores import BaseVectorStore, PointData, PointPayload
from .document_chunker import chunk_document
from .document_embedder import embed_chunks

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _trigrams(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else set()


def _jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicate_chunks(chunks: list[Chunk], threshold: float) -> list[Chunk]:
    """Remove near-duplicate chunks using trigram Jaccard similarity.

    The first occurrence of any near-duplicate group is kept.
    Comparison is performed on normalized (lowercased, collapsed-whitespace)
    content strings. O(n²) — acceptable for typical document sizes.
    """
    accepted: list[Chunk] = []
    accepted_normalized: list[str] = []

    for chunk in chunks:
        norm = _normalize_for_dedup(chunk.content)
        duplicate = any(
            _jaccard(norm, existing) >= threshold for existing in accepted_normalized
        )
        if not duplicate:
            accepted.append(chunk)
            accepted_normalized.append(norm)

    return accepted


def _build_points(
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[PointData]:
    """Zip chunks + vectors into provider-agnostic points."""
    if len(chunks) != len(vectors):
        raise IndexingError(
            f"Vector/chunk count mismatch: {len(vectors)} vectors for "
            f"{len(chunks)} chunks."
        )

    return [
        PointData(
            id=uuid4(),
            vector=vector,
            payload=PointPayload(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                source_type=chunk.source_type.value,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            ),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


async def index_document(
    document_id: UUID,
    upload_dir: Path,
    provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    collection_name: str,
    distance: DistanceMetric,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int,
    min_chunk_tokens: int,
    dedup_threshold: float,
) -> IndexingResponse:
    """Parse → Chunk → Deduplicate → Embed → Upsert pipeline.

    Pure orchestration: this function never touches Qdrant, HTTP, or
    FastAPI. Vendor concerns stay inside ``vector_store``; transport
    concerns stay inside the API layer.
    """
    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=upload_dir,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
    )

    if not chunks:
        raise IndexingError(f"Document '{document_id}' produced no chunks to index.")

    chunks = _deduplicate_chunks(chunks, threshold=dedup_threshold)

    if not chunks:
        raise IndexingError(
            f"Document '{document_id}' produced no unique chunks after deduplication."
        )

    vectors = await embed_chunks(chunks, provider)
    points = _build_points(chunks, vectors)
    dimension = provider.embedding_dimension

    await vector_store.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        distance=distance,
    )

    await vector_store.delete_by_document(
        collection_name=collection_name,
        document_id=str(document_id),
    )

    indexed = await vector_store.upsert(
        collection_name=collection_name,
        points=points,
    )
    return IndexingResponse(
        document_id=document_id,
        collection_name=collection_name,
        total_chunks=len(chunks),
        indexed_points=indexed,
        embedding_model=provider.model_name,
        dimension=dimension,
        status=DocumentStatus.INDEXED,
    )
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/services/test_document_indexer.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 7.5: Run full test suite**

```bash
cd backend
python -m pytest -v
```

- [ ] **Step 7.6: Commit**

```bash
git add app/services/document_indexer.py tests/services/test_document_indexer.py
git commit -m "feat(indexer): add deduplication gate and enrich PointPayload"
```

---

## Task 8 — API Layer: Thread `min_chunk_tokens` and `dedup_threshold` Through

**Files:**
- Modify: `app/api/v1/index.py`
- Modify: `app/api/v1/embed.py`
- Modify: `app/services/document_embedder.py`
- Create: `tests/api/__init__.py`, `tests/api/test_index_endpoint_smoke.py`

**Interfaces:**
- Consumes: `settings.MIN_CHUNK_TOKENS` and `settings.DEDUP_SIMILARITY_THRESHOLD` (Task 2).
- Produces: `/index` and `/embed` pass the new config values through to their respective service functions. No new endpoint parameters; no response schema changes.

---

- [ ] **Step 8.1: Write the smoke test**

```python
# tests/api/test_index_endpoint_smoke.py
"""Smoke test: /index route exists and is wired after the Task 8 changes."""
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app


def test_index_route_is_registered():
    client = TestClient(app)
    # A 404 means the route is missing. Any other status is acceptable here.
    response = client.post(f"/api/v1/documents/{uuid4()}/index")
    assert response.status_code != 404
```

- [ ] **Step 8.2: Run to verify smoke test passes (baseline — route exists)**

```bash
cd backend
python -m pytest tests/api/test_index_endpoint_smoke.py -v
```

Expected: Passes before any changes (route already exists).

- [ ] **Step 8.3: Update `app/api/v1/index.py`**

  Pass the two new settings fields to `index_document`:
  ```python
  return await index_document(
      document_id=document_id,
      upload_dir=settings.UPLOAD_DIR,
      provider=provider,
      vector_store=vector_store,
      collection_name=settings.QDRANT_COLLECTION,
      distance=settings.DISTANCE_METRIC,
      strategy=ChunkingStrategy.TOKEN,
      chunk_size=settings.DEFAULT_CHUNK_SIZE,
      overlap=settings.DEFAULT_CHUNK_OVERLAP,
      min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
      dedup_threshold=settings.DEDUP_SIMILARITY_THRESHOLD,
  )
  ```

- [ ] **Step 8.4: Update `app/services/document_embedder.py`**

  Add `min_chunk_tokens: int` to `embed_document` and pass it to `chunk_document`:
  ```python
  async def embed_document(
      document_id: UUID,
      upload_dir: Path,
      provider: BaseEmbeddingProvider,
      strategy: ChunkingStrategy,
      chunk_size: int,
      overlap: int,
      min_chunk_tokens: int,   # ← new
  ) -> EmbeddingResponse:
      """Parse → Chunk → Embed pipeline. Stateless: nothing is persisted here."""
      chunks = await chunk_document(
          document_id=document_id,
          upload_dir=upload_dir,
          strategy=strategy,
          chunk_size=chunk_size,
          overlap=overlap,
          min_chunk_tokens=min_chunk_tokens,  # ← new
      )
      await embed_chunks(chunks, provider)
      return EmbeddingResponse(
          document_id=document_id,
          total_chunks=len(chunks),
          embedding_model=provider.model_name,
          dimension=provider.embedding_dimension,
          status=DocumentStatus.EMBEDDING,
      )
  ```

- [ ] **Step 8.5: Update `app/api/v1/embed.py`**

  Pass `min_chunk_tokens` to `embed_document`:
  ```python
  return await embed_document(
      document_id=document_id,
      upload_dir=settings.UPLOAD_DIR,
      provider=provider,
      strategy=ChunkingStrategy.TOKEN,
      chunk_size=settings.DEFAULT_CHUNK_SIZE,
      overlap=settings.DEFAULT_CHUNK_OVERLAP,
      min_chunk_tokens=settings.MIN_CHUNK_TOKENS,
  )
  ```

- [ ] **Step 8.6: Run full test suite**

```bash
cd backend
python -m pytest -v
```

Expected: All tests pass with no regressions.

- [ ] **Step 8.7: Commit**

```bash
git add app/api/v1/index.py app/api/v1/embed.py \
        app/services/document_embedder.py \
        tests/api/__init__.py tests/api/test_index_endpoint_smoke.py
git commit -m "feat(api): thread min_chunk_tokens and dedup_threshold through index and embed routes"
```

---

## Acceptance Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| AC-1 | `/index` response schema is unchanged (`IndexingResponse` fields identical) | Existing contract preserved; compare schema before/after |
| AC-2 | Chunks below `MIN_CHUNK_TOKENS` words are merged, not stored standalone | `test_tiny_chunk_is_merged_into_neighbor` passes |
| AC-3 | Empty or whitespace-only chunks never reach the embedder | `test_chunk_document_all_empty_pages_returns_empty` + defensive `content.strip()` guard |
| AC-4 | Every `Chunk` carries all 7 metadata fields: `document_id`, `chunk_id`, `chunk_index`, `page_number`, `start_char`, `end_char`, `source_type` | `test_chunk_accepts_source_type` + existing `Chunk` model validation |
| AC-5 | Text is normalized before chunking (CRLF→LF, repeated whitespace, excessive blank lines) | `test_normalizer.*` suite passes |
| AC-6 | Near-duplicate chunks (trigram Jaccard ≥ `DEDUP_SIMILARITY_THRESHOLD`) are dropped before indexing | `test_deduplicate_removes_near_identical_chunks` passes |
| AC-7 | Distinct chunks are never filtered | `test_deduplicate_keeps_distinct_chunks` passes |
| AC-8 | `PointPayload` in Qdrant includes `source_type`, `start_char`, `end_char` | `test_build_points_populates_source_type_and_offsets` passes |
| AC-9 | `/embed` endpoint continues to work | `test_index_route_is_registered` + manual smoke |
| AC-10 | Services remain stateless | Code review: no module-level mutable state, no cross-request state |
| AC-11 | No caching introduced | Code review: no `functools.cache`/`lru_cache` on any service function |

---

## Testing Strategy

### Unit tests (no I/O, no network)

| Test file | Coverage |
|-----------|----------|
| `tests/chunkers/test_normalizer.py` | All 9 normalization rules in isolation |
| `tests/chunkers/test_models.py` | `ChunkingConfig` and `ChunkSpan` validation with new fields |
| `tests/chunkers/test_token_chunker.py` | Normalization + merge pass in `RecursiveChunker` |
| `tests/core/test_config.py` | New settings fields and defaults |
| `tests/schemas/test_chunk_schema.py` | `Chunk` schema with `source_type` |
| `tests/vectorstores/test_point_payload.py` | `PointPayload` with new fields |
| `tests/services/test_document_chunker.py` | `chunk_document` with mocked parser (empty pages, source_type) |
| `tests/services/test_document_indexer.py` | `_deduplicate_chunks` + `_build_points` unit tests |

### Smoke tests

| Test file | Coverage |
|-----------|----------|
| `tests/api/test_index_endpoint_smoke.py` | Route registration is intact after wiring |

### Manual regression check (before merging)

1. Start Qdrant: `docker compose up -d`
2. Upload a PDF with repeated headers/footers (e.g. a report with a site-wide header on every page).
3. Call `POST /api/v1/documents/{id}/index`. Inspect `total_chunks` — should be smaller than before deduplication was added.
4. Inspect a point directly in Qdrant (`GET /collections/documents/points/{id}`) — payload must include `source_type`, `start_char`, `end_char`.
5. Upload and re-index the same document. Confirm point count is stable (idempotent — old points are deleted before upserting).

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Merge pass changes chunk boundaries, making `start_char`/`end_char` slightly off on merged chunks | Medium | Use `text.find(merged_text, search_start)` on the post-normalized text, same as existing code. Cover with `test_span_offsets_are_within_original_text_bounds`. |
| Trigram Jaccard silently drops legitimate near-similar (but distinct) content | Low | Default threshold of 0.97 is deliberately conservative. Tune via `DEDUP_SIMILARITY_THRESHOLD` env var. Cover with `test_deduplicate_keeps_distinct_chunks`. |
| New required fields on `ChunkingConfig` break the `/chunks` endpoint (which builds config indirectly via `chunk_document`) | High | Task 6 step 6.5 runs the full test suite before committing; fix any breakage before proceeding. |
| `PointPayload` field additions are backward-incompatible with existing Qdrant collection data | Medium | Qdrant uses dynamic payload schemas; new fields on new inserts are fine. Old points in existing collections will lack the new keys — treat those collections as stale and re-index. |
| `normalize_text` collapses intentional whitespace in code blocks inside Markdown | Low | Normalization is applied at the chunker level, not the parser level. For Markdown, each section is already split by `MdParser` before `RecursiveChunker` sees it. Revisit in P3 if code-block quality is impacted. |

---

## Re-indexing Note

This phase modifies chunking logic (`RecursiveChunker`), metadata (`Chunk` fields), and what gets stored (`PointPayload`). Per the checklist policy:

- All existing Qdrant collections indexed before this change are **stale**.
- They must be rebuilt with `/index` after deploying Phase 1.
- The re-indexing decision (rebuild, migrate, or retire old collections) must be made explicit before deploying to production.
