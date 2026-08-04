# Phase 2: Step 10 Implementation Plan

> **Scope:** Separate storage chunk size from LLM prompt chunk size (Model & Config only).
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Disambiguate the concept of a single `chunk_size` into two distinct concepts: `embedding_chunk_size` (for vector matching) and `prompt_chunk_size` (for LLM context). This step implements only the minimal schema, configuration, and validation changes needed to represent this separation, without altering the underlying chunking algorithm yet.

**Architecture:**
- `app/core/config.py` drops `DEFAULT_CHUNK_SIZE` in favor of `DEFAULT_EMBEDDING_CHUNK_SIZE` and `DEFAULT_PROMPT_CHUNK_SIZE`.
- API boundaries (`ChunkRequest`, `index_document`, `embed_document`) are updated to accept both sets of sizes.
- `ChunkingConfig` holds both sets.
- The `max_sequence_length` guardrail is updated to strictly target `embedding_chunk_size`.
- The actual chunker algorithm remains untouched (temporarily ignoring `prompt_chunk_size` until a later step introduces hierarchical chunking).

**Tech Stack:** Python 3.12, Pydantic, pytest

---

## Global Constraints
- Do not implement two-pass hierarchical chunking.
- Do not duplicate `parent_content` or alter `Chunk`/`PointPayload` definitions.
- Do not touch retrieval behavior or ranking measurement.
- Ensure all validations strictly enforce `embedding_chunk_size <= prompt_chunk_size`.
- Ensure the `max_sequence_length` guardrail explicitly targets `embedding_chunk_size`.

---

### Task 1: Update Application Settings

**Files:**
- Modify: `app/core/config.py`
- Modify: `backend/.env`
- Modify: `tests/core/test_config.py`

- [ ] **Step 1: Write the failing tests (RED)**
Update `tests/core/test_config.py`. Remove `DEFAULT_CHUNK_SIZE`. Add tests asserting `DEFAULT_EMBEDDING_CHUNK_SIZE` is 500, `DEFAULT_EMBEDDING_OVERLAP` is 50, and introducing `DEFAULT_PROMPT_CHUNK_SIZE` (e.g., 2000) and `DEFAULT_PROMPT_OVERLAP` (e.g., 200). Add a test asserting validation fails if `embedding_chunk_size > prompt_chunk_size`.

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write minimal implementation (GREEN)**
In `config.py`, replace the chunking fields with the new separated fields. Add a `@model_validator` to ensure `DEFAULT_EMBEDDING_CHUNK_SIZE <= DEFAULT_PROMPT_CHUNK_SIZE`. Update `.env` to match.

- [ ] **Step 4: Run test to verify it passes**

---

### Task 2: Update Domain Models and API Schemas

**Files:**
- Modify: `app/schemas/chunk.py`
- Modify: `app/chunkers/models.py`
- Modify: `tests/schemas/test_chunk_schema.py`
- Modify: `tests/chunkers/test_models.py`

- [ ] **Step 5: Write the failing tests (RED)**
Update schema tests to check for `embedding_chunk_size` and `prompt_chunk_size` on `ChunkRequest` and `ChunkingConfig`. Add validation tests ensuring overlaps are valid for their respective chunk sizes, and that embedding size <= prompt size.

- [ ] **Step 6: Run test to verify it fails**

- [ ] **Step 7: Write minimal implementation (GREEN)**
Update `ChunkRequest` and `ChunkingConfig`. Rename `chunk_size` to `embedding_chunk_size` and `overlap` to `embedding_overlap`. Add `prompt_chunk_size` and `prompt_overlap`. Update the Pydantic validators.

- [ ] **Step 8: Run test to verify it passes**

---

### Task 3: Update Services and Guardrails

**Files:**
- Modify: `app/services/document_indexer.py`
- Modify: `app/services/document_embedder.py`
- Modify: `app/services/document_chunker.py`
- Modify: `app/chunkers/token.py` (update to use config.embedding_chunk_size)
- Modify: Test files for all the above services

- [ ] **Step 9: Write the failing tests (RED)**
Update `test_document_indexer.py` and `test_document_embedder.py` to pass the new keyword arguments. Update the guardrail test to assert that `EmbeddingError`/`IndexingError` is raised if `embedding_chunk_size > provider.max_sequence_length`.

- [ ] **Step 10: Run test to verify it fails**

- [ ] **Step 11: Write minimal implementation (GREEN)**
Refactor the service signatures to accept the separated parameters.
Update the guardrail in the indexer and embedder to check `embedding_chunk_size`.
In `chunk_document`, pass the separated parameters to `ChunkingConfig`.
In `token.py`, update `RecursiveCharacterTextSplitter` to use `config.embedding_chunk_size` (satisfying the constraint to not implement two-pass chunking yet).

- [ ] **Step 12: Run test to verify it passes**

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
Commit message: `refactor: separate embedding chunk size from prompt chunk size in models and config`

## Verification Checklist
- [ ] `config.py` and `.env` strictly separate embedding vs prompt sizing.
- [ ] `ChunkRequest` and `ChunkingConfig` represent both concepts.
- [ ] Cross-field validation prevents embedding size from exceeding prompt size.
- [ ] Service guardrails strictly evaluate the embedding size against the model limit.
- [ ] The core chunking algorithm remains a single pass (using embedding size for now).
