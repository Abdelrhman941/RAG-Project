# Phase 1 — Step 1: Text Normalization Module

> **Status:** awaiting review — no code written yet.

---

## Blockers Fixed (vs. original plan)

Four issues in the original plan are corrected here before any code is written.

### 1. `MIN_CHUNK_TOKENS` naming vs. merge logic mismatch

The original plan named the threshold `MIN_CHUNK_TOKENS` and the setting held an
**integer token count**. The merge logic then measured token count by
`len(chunk.split())` (whitespace-split word count). These are not the same thing:
tiktoken token count ≠ whitespace-split word count for most real text.

**Fix applied in this plan:**

- Rename the concept to `MIN_CHUNK_CHARS` (a character count) and rename the
  setting to `MIN_CHUNK_CHARS` accordingly.
- The merge/drop test in `RecursiveChunker._merge_small_chunks` becomes
  `len(chunk.strip()) < min_chunk_chars` — a pure character count, which is
  deterministic, has no external dependency, and exactly matches what
  `ChunkSpan.char_count` already records.
- This keeps the config field and the merge gate in the same unit, eliminating
  the mismatch.

> **Note:** `MIN_CHUNK_CHARS` is introduced in Step 1 (config) and consumed in
> Step 4 (RecursiveChunker) — Step 1 itself does not use it, but naming is
> settled here so all later steps are consistent.

### 2. Unsafe merged-chunk offset strategy

The original plan recomputed `start_char`/`end_char` after merging by calling
`text.find(merged_content, search_start)`. After a merge, `merged_content` is
`chunk_a + "\n" + chunk_b`. The merged string may not appear verbatim in the
normalized text (overlap causes `chunk_b` to start before `chunk_a` ends, so the
concatenation is synthetic). `text.find` can return `-1` and then falls back to
`search_start`, silently producing wrong offsets.

**Fix applied in this plan:**

- Offsets are tracked **before merging**, at the raw-chunk level.
- `_split_with_offsets(text, splitter)` returns `list[tuple[str, int, int]]`
  (content, start, end) by scanning the normalized text once with
  `text.find(chunk, search_start)` — exactly as the current code does, but
  **before** any merge.
- The merge pass then operates on `(content, start, end)` tuples:
  merging two adjacent tuples produces a new tuple whose `start` is the first
  tuple's `start` and whose `end` is the second tuple's `end`. No re-search
  needed.
- If `text.find` returns `-1` for a raw (pre-merge) chunk, the chunk is logged
  and dropped — it is never silently given a wrong offset.

### 3. Small-document content loss

The original `_merge_small_chunks` drop rule discarded any chunk still below the
threshold after the two-pass merge. For a document whose entire content fits into
a single short chunk (e.g. a two-sentence abstract), this would drop the only
chunk and `index_document` would raise `IndexingError` even though the document
had real content.

**Fix applied in this plan:**

- The merge pass keeps a chunk that is still small **only if it is the sole
  remaining chunk** after merging. The rule: if after merging the result list
  has exactly one entry and its char count is below the threshold, keep it
  anyway — it represents the entire document content.
- This preserves the intent (prevent micro-fragments from reaching the vector
  store) while never discarding the only meaningful content a document has.

### 4. Architecture wording

The original architecture line read "All changes are confined to the service and
chunker layers." That was inaccurate — the plan also modifies the config, core
enums, schemas, vectorstore models, and API routes.

**Fix applied in this plan:**

The corrected architecture statement (used in the revised full-plan header) is:

> "The primary work is in the chunker and service layers. Supporting changes
> touch config (two new settings), core enums (SourceType), data models
> (ChunkSpan, ChunkingConfig, Chunk, PointPayload), and the API routes
> (/index, /embed) to thread the new settings through. No new endpoints.
> No response schema changes."

---

## Step 1 — Revised Plan: Text Normalization Module

### Objective

Create a single, pure, stateless function `normalize_text(text: str) -> str`
inside a new module `app/chunkers/normalizer.py`. This function is the sole
normalization point for all page text before it reaches any chunking strategy.
It is called by `RecursiveChunker.chunk()` in Step 4.

**In scope:**

- Normalize CRLF and bare CR to LF.
- Collapse runs of horizontal whitespace (spaces, tabs) within a line to a
  single space, without touching newlines.
- Collapse three or more consecutive newlines to exactly two (one blank line).
- Strip leading and trailing whitespace from the final result.

**Out of scope for this step:**

- Header/footer detection (P3 item per checklist).
- Deduplication (Step 7).
- Any merging or filtering logic (Step 4).
- Any change to `__init__.py`, config, enums, or schemas.

---

### Files

| Action | Path |
|--------|------|
| Create | `app/chunkers/normalizer.py` |
| Create | `tests/__init__.py` *(package marker — empty)* |
| Create | `tests/chunkers/__init__.py` *(package marker — empty)* |
| Create | `tests/chunkers/test_normalizer.py` |

`app/chunkers/__init__.py` is **not** modified in this step. The export is added
in Step 4 when `RecursiveChunker` imports it, so the public API stays in sync
with actual consumers.

---

### Test-First Breakdown (TDD)

Tests are written in exactly this order. Each test must be watched to **fail
for the right reason** before any implementation code exists.

#### RED — write all failing tests first

```python
# tests/chunkers/test_normalizer.py
"""Unit tests for app.chunkers.normalizer.normalize_text.

Each test covers one transformation rule in isolation.
No mocks. No I/O. Pure function.
"""

import pytest

from app.chunkers.normalizer import normalize_text


# ── Line-ending normalization ──────────────────────────────────────────────

def test_crlf_is_replaced_by_lf():
    assert normalize_text("a\r\nb") == "a\nb"


def test_bare_cr_is_replaced_by_lf():
    assert normalize_text("a\rb") == "a\nb"


def test_mixed_crlf_and_cr_both_normalized():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


# ── Horizontal whitespace normalization ───────────────────────────────────

def test_multiple_spaces_collapsed_to_one():
    assert normalize_text("hello   world") == "hello world"


def test_tab_replaced_by_single_space():
    assert normalize_text("hello\tworld") == "hello world"


def test_mixed_spaces_and_tabs_collapsed():
    assert normalize_text("a \t b") == "a b"


def test_newlines_not_touched_by_whitespace_collapse():
    # The whitespace collapse must not eat newlines.
    assert normalize_text("a\n b") == "a\n b"


# ── Blank-line normalization ───────────────────────────────────────────────

def test_three_newlines_collapsed_to_two():
    assert normalize_text("a\n\n\nb") == "a\n\nb"


def test_many_newlines_collapsed_to_two():
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_exactly_two_newlines_unchanged():
    # Two newlines (one blank line) are already canonical — must not be touched.
    assert normalize_text("a\n\nb") == "a\n\nb"


# ── Leading / trailing whitespace ─────────────────────────────────────────

def test_leading_spaces_stripped():
    assert normalize_text("  hello") == "hello"


def test_trailing_spaces_stripped():
    assert normalize_text("hello  ") == "hello"


def test_leading_newline_stripped():
    assert normalize_text("\nhello") == "hello"


# ── Edge cases ────────────────────────────────────────────────────────────

def test_empty_string_returns_empty_string():
    assert normalize_text("") == ""


def test_only_whitespace_returns_empty_string():
    assert normalize_text("   \n\t  ") == ""


def test_already_normalized_text_is_unchanged():
    text = "First paragraph.\n\nSecond paragraph."
    assert normalize_text(text) == text


def test_single_word_unchanged():
    assert normalize_text("hello") == "hello"
```

**Run before writing a single line of implementation:**

```bash
cd backend
python -m pytest tests/chunkers/test_normalizer.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'app.chunkers.normalizer'`

If any test **passes** at this stage, that test is wrong — fix it before
implementing.

---

#### GREEN — minimal implementation

Write only what is needed to pass the tests above. No extra transformations.

```python
# app/chunkers/normalizer.py
"""Pure text normalization for the ingestion pipeline.

This module provides a single public function, ``normalize_text``,
that is called once per page before any chunking strategy is applied.
All transformations are deterministic and stateless — the function has
no side effects and no dependencies on external state.
"""

import re

# Matches three or more consecutive newline characters.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Matches one or more horizontal whitespace characters (spaces and tabs)
# that do NOT include newlines.  Used to collapse inline spacing without
# touching line structure.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Return a normalized copy of *text* ready for chunking.

    Transformations applied in order:

    1. **Line endings** — ``\\r\\n`` and bare ``\\r`` → ``\\n``.
    2. **Inline whitespace** — runs of spaces/tabs within a line → single space.
    3. **Excessive blank lines** — three or more consecutive ``\\n`` → ``\\n\\n``.
    4. **Strip** — remove leading and trailing whitespace.

    An empty string or an all-whitespace string returns ``""``.
    """
    if not text:
        return text

    # 1. Normalize line endings (CRLF before bare CR to avoid double-replace).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. Collapse inline whitespace — regex skips newlines.
    text = _INLINE_WS_RE.sub(" ", text)
    # 3. Collapse excessive blank lines.
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    # 4. Strip.
    return text.strip()
```

**Verify GREEN:**

```bash
cd backend
python -m pytest tests/chunkers/test_normalizer.py -v
```

Expected: all 17 tests pass, no warnings.

---

#### REFACTOR

The implementation above has no duplication to remove. No refactor needed.
Check the output is clean (no warnings, no deprecations) before committing.

---

### Commit

```bash
git add app/chunkers/normalizer.py \
        tests/__init__.py \
        tests/chunkers/__init__.py \
        tests/chunkers/test_normalizer.py
git commit -m "feat(chunker): add pure text normalizer (normalize_text)"
```

---

### Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `_INLINE_WS_RE` eats newlines and flattens multi-line text into a single line | Medium | High | `[^\S\n]+` explicitly excludes `\n`. Test `test_newlines_not_touched_by_whitespace_collapse` catches this directly. |
| R2 | CRLF replacement order matters — doing bare `\r` first would convert `\r\n` to `\n\n` | Low | Medium | The implementation replaces `\r\n` **before** bare `\r`. Test `test_crlf_is_replaced_by_lf` and `test_mixed_crlf_and_cr_both_normalized` both verify the correct result. |
| R3 | `text.strip()` after collapsing blank lines removes trailing newlines that downstream code might expect | Low | Low | The chunker in Step 4 operates on normalized text; trailing newlines carry no semantic value for RAG chunking. Tests `test_trailing_spaces_stripped` and `test_leading_newline_stripped` confirm strip behavior is intentional. |
| R4 | `normalize_text` is not yet exported from `app/chunkers/__init__.py`, so a consumer importing from the package will fail | Low | Low | Intentional for this step — the export is added in Step 4 alongside the consumer (`RecursiveChunker`). No consumer exists yet. |

---

### Acceptance Criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| AC-1 | `normalize_text("a\r\nb")` returns `"a\nb"` | `test_crlf_is_replaced_by_lf` |
| AC-2 | `normalize_text("a\rb")` returns `"a\nb"` | `test_bare_cr_is_replaced_by_lf` |
| AC-3 | `normalize_text("a \t b")` returns `"a b"` | `test_mixed_spaces_and_tabs_collapsed` |
| AC-4 | Newlines are **not** eaten by the inline whitespace collapse | `test_newlines_not_touched_by_whitespace_collapse` |
| AC-5 | `normalize_text("a\n\n\nb")` returns `"a\n\nb"` | `test_three_newlines_collapsed_to_two` |
| AC-6 | `normalize_text("a\n\nb")` returns `"a\n\nb"` unchanged | `test_exactly_two_newlines_unchanged` |
| AC-7 | `normalize_text("")` returns `""` | `test_empty_string_returns_empty_string` |
| AC-8 | `normalize_text("   \n\t  ")` returns `""` | `test_only_whitespace_returns_empty_string` |
| AC-9 | Already-normalized text is returned unchanged | `test_already_normalized_text_is_unchanged` |
| AC-10 | All 17 tests pass; `python -m pytest tests/chunkers/test_normalizer.py -v` exits 0 | CI / manual run |
| AC-11 | `normalizer.py` has no imports beyond the standard library | Code review |
| AC-12 | `normalize_text` has no side effects, no mutable module-level state | Code review |

---

## What Comes Next (not in scope here)

Once this step is reviewed and approved, Step 2 will cover:

- `MIN_CHUNK_CHARS` and `DEDUP_SIMILARITY_THRESHOLD` added to `app/core/config.py`.
- The naming fix (`MIN_CHUNK_CHARS` not `MIN_CHUNK_TOKENS`) is the first thing
  the config step will establish, making it the authoritative source for all
  later steps.
