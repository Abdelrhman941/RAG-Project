# Phase 2: Step 9 Implementation Plan

> **Scope:** Add a guardrail for maximum chunk size before embedding.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the requested `chunk_size` never exceeds the embedding model's context window (`max_sequence_length`). If it does, fail fast instead of silently truncating text and destroying retrieval quality.

**Why this step should come first:**
We cannot safely tune chunk sizes (Item 3) or build complex small-to-big retrieval hierarchies (Items 5-7) if the embedder is silently dropping text. Establishing a hard architectural boundary guarantees data integrity as we scale.

**Architecture:**
- `BaseEmbeddingProvider` gains an abstract `max_sequence_length` property.
- `SentenceTransformerProvider` exposes the model's actual `max_seq_length`.
- `index_document` and `embed_document` validate `chunk_size <= provider.max_sequence_length` before doing any work.
- `app/core/config.py` defaults are lowered to fit within the `multilingual-e5-small` 512-token limit, ensuring the app works out of the box without hitting the new guardrail.

**Tech Stack:** Python 3.12, Pydantic, pytest, SentenceTransformers

---

## Global Constraints
- Keep the service layer vendor-agnostic (rely only on `BaseEmbeddingProvider`).
- Do not introduce embedding-based deduplication or caching.
- Do not touch other P2 items (like hierarchical chunks).
- Treat `max_sequence_length` as the absolute ceiling for `chunk_size`.

---

### Task 1: Expose `max_sequence_length` on the Provider

**Files:**
- Modify: `app/embedders/base.py`
- Modify: `app/embedders/sentence_transformer.py`
- Create/Modify: `tests/embedders/test_sentence_transformer.py` (if missing, create it to test the provider)

- [ ] **Step 1: Write the failing tests (RED)**
Create `tests/embedders/test_sentence_transformer.py` asserting that `max_sequence_length` is exposed as an integer, and that `BaseEmbeddingProvider` requires it.

- [ ] **Step 2: Run test to verify it fails**
Run `uv run pytest tests/embedders/test_sentence_transformer.py -v`. Expected: `AttributeError` or similar.

- [ ] **Step 3: Write minimal implementation (GREEN)**
In `app/embedders/base.py`:
```python
@property
@abstractmethod
def max_sequence_length(self) -> int:
    """Maximum number of tokens the model can process."""
    raise NotImplementedError
```
In `app/embedders/sentence_transformer.py`:
```python
@property
def max_sequence_length(self) -> int:
    seq_len = self._model.max_seq_length
    if seq_len is None:
        raise EmbeddingError(f"Model '{self._model_name}' does not expose max_seq_length.")
    return int(seq_len)
```

- [ ] **Step 4: Run test to verify it passes**
Run `uv run pytest tests/embedders/test_sentence_transformer.py -v`. Expected: PASS.

---

### Task 2: Fix Default Config Regression

**Hidden Dependency:** Our current default `chunk_size` is 1000. `multilingual-e5-small` has a context window of 512. The moment we add the guardrail in Task 3, the app will break out of the box. We must fix the defaults first.

**Files:**
- Modify: `app/core/config.py`
- Modify: `tests/core/test_config.py`

- [ ] **Step 5: Write the failing tests (RED)**
Update `tests/core/test_config.py` to assert `DEFAULT_CHUNK_SIZE` is 500 and `DEFAULT_CHUNK_OVERLAP` is 50.

- [ ] **Step 6: Run test to verify it fails**
Run `uv run pytest tests/core/test_config.py -v`.

- [ ] **Step 7: Write minimal implementation (GREEN)**
In `app/core/config.py`:
```python
DEFAULT_CHUNK_SIZE: int = Field(default=500, gt=0)
DEFAULT_CHUNK_OVERLAP: int = Field(default=50, ge=0)
```

- [ ] **Step 8: Run test to verify it passes**
Run `uv run pytest tests/core/test_config.py -v`.

---

### Task 3: Enforce the Guardrail in Services

**Files:**
- Modify: `app/services/document_indexer.py`
- Modify: `app/services/document_embedder.py`
- Modify: `tests/services/test_document_indexer.py`
- Modify: `tests/services/test_document_embedder.py`

- [ ] **Step 9: Write the failing tests (RED)**
Add tests asserting `IndexingError` and `EmbeddingError` are raised when `chunk_size > provider.max_sequence_length`.

- [ ] **Step 10: Run test to verify it fails**
Run `uv run pytest tests/services/test_document_indexer.py tests/services/test_document_embedder.py -v`.

- [ ] **Step 11: Write minimal implementation (GREEN)**
In both `index_document` and `embed_document`, before chunking:
```python
if chunk_size > provider.max_sequence_length:
    raise IndexingError( # or EmbeddingError
        f"Requested chunk_size ({chunk_size}) exceeds model maximum context length ({provider.max_sequence_length})."
    )
```

- [ ] **Step 12: Run test to verify it passes**
Run the tests again. Expected: PASS.

---

### Task 4: Full Verification

- [ ] **Step 13: Full Suite**
Run:
```bash
uv run ruff check .
uv run mypy .
uv run pytest -v
```

- [ ] **Step 14: Commit**
Commit message: `feat: enforce max_sequence_length guardrail for chunk size`

## Risks and Regression Impact
- **Risk:** Models without a clear `max_seq_length` attribute might break. 
  - *Mitigation:* `SentenceTransformer` universally sets this attribute (sometimes defaulting to 512 if undefined by the tokenizer). We raise a clear `EmbeddingError` if it is missing, making debugging easy.
- **Regression:** Out-of-the-box config is safely lowered. Existing users with custom `.env` files setting `DEFAULT_CHUNK_SIZE=1000` will encounter startup/indexing errors until they lower their size or swap their model. This is the intended behavior (failing fast instead of silently destroying data).

## Verification Checklist
- [ ] `max_sequence_length` is abstract on base provider.
- [ ] Service endpoints fail fast if `chunk_size` exceeds model limits.
- [ ] Defaults adjusted to fit `multilingual-e5-small`.
- [ ] Full suite passes cleanly.
