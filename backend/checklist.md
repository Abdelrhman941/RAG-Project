# RAG Refactor Checklist

## Decision

* `/index` is the production ingestion path: **parse → chunk → embed → upsert**.
* `/embed` is an optional inspection/debug endpoint.
* Do **not** design the pipeline around calling `/embed` before `/index`.
* Keep services stateless; keep system state in persistent stores (uploads, vector DB, metadata DB), not in service memory.

---

## P0 — Baseline and safety

* [x] Preserve the current endpoint contracts before refactoring.
* [ ] Record baseline outputs for each endpoint on a fixed sample corpus.
* [ ] Measure current latency for parse, chunk, embed, index, retrieve, and chat.
* [ ] Log chunk counts, chunk sizes, and retrieval result counts.
* [x] Make sure test fixtures and imports do not depend on pre-refactor names.
* [x] Keep the data flow explicit for each document lifecycle stage.

---

## P1 — Ingestion pipeline fixes

* [x] Remove any duplicated compute between endpoints and services.
* [x] Ensure the indexing pipeline is the single source of truth for ingestion.
* [ ] Add a lightweight cache only if it is truly necessary and explicitly designed.
* [x] Add `min_chunk_size` or `min_chunk_tokens`.
* [x] Merge very small chunks with neighboring chunks instead of storing them alone.
* [x] Prevent empty or near-empty chunks from reaching the vector store.
* [x] Add clearer metadata for every chunk:

  * [x] `document_id`
  * [x] `chunk_id`
  * [x] `chunk_index`
  * [x] `page_number`
  * [x] `start_char`
  * [x] `end_char`
  * [x] `source_type`
* [ ] Normalize text before chunking:

  * [x] line endings
  * [x] repeated whitespace
  * [x] empty sections
  * [ ] repeated headers and footers when possible
* [x] Deduplicate highly similar repeated content.
* [x] Prevent repeated or near-duplicate content from being indexed multiple times.

---

## P2 — Chunk quality improvements

* [x] Make chunking operate on normalized text, not file shape alone.
* [x] Use one consistent chunking policy for all text-based inputs.
* [ ] Tune `chunk_size` and `overlap` using a real corpus, not guesses.
* [x] Add a guardrail for maximum chunk size before embedding.
* [x] Separate storage chunk size from LLM prompt chunk size when needed.
* [x] Add parent or section references for each chunk.
* [x] Support small-to-big retrieval:

  * [x] small chunk for matching
  * [x] larger parent section for prompt context
* [ ] Measure the effect of micro-chunks on ranking quality.

---

## P3 — Parser upgrades

* [x] Add table support to `PdfParser`.
* [x] Add OCR support for text inside images.
* [ ] Add layout detection support.
* [ ] Add figure extraction support.
* [ ] Add citation mapping support.
* [ ] Improve Markdown parsing so headings do not create contextless chunks.
* [ ] Improve TXT parsing so large files are split into useful chunks.
* [ ] Detect repeated slide build-ups in presentation-style PDFs.

---

## P4 — Retrieval power

* [x] Add hybrid search: vector search + keyword search.
* [x] Merge results with RRF or a similar fusion strategy.
* [ ] Add query transformation before retrieval.
* [ ] Add multi-query retrieval when the user question is ambiguous.
* [x] Add reranking for retrieved results.
* [x] Filter out very weak results before prompt construction.
* [x] Reduce duplicate or near-duplicate chunks in `top_k`.
* [ ] Add parent document or section retrieval.
* [ ] Measure false positives caused by micro-chunks.

---

## P5 — Performance and cost

* [x] Batch embeddings instead of embedding one chunk at a time when possible.
* [x] Reduce unnecessary round trips to the vector store.
* [ ] Cache embeddings only if the cache design is explicit and safe.
* [ ] Cache retrieval results only if the query pattern justifies it.
* [x] Use async execution where it materially improves throughput.
* [x] Avoid recreating splitters or helpers inside tight loops.
* [ ] Benchmark every major change before and after.
* [ ] Keep a stable benchmark corpus for regression testing.

---

## P6 — Reliability and quality gates

* [ ] Create golden samples for PDF, Markdown, and TXT ingestion.
* [ ] Add checks for chunk count, average chunk size, and duplicate ratio.
* [x] Add checks for empty chunks or tiny chunks.
* [ ] Add checks for text loss during parsing.
* [x] Add checks that the embedder input stays within limits.
* [ ] Add checks for citation mapping correctness.
* [ ] Verify grounded answers in `/chat`.
* [ ] Reduce hallucination risk through retrieval constraints and prompt rules.
* [ ] Add regression tests for every fixed bug.

---

## P7 — Post E2E v1 upgrades

* [ ] PDF table support.
* [ ] Better-than-character chunking.
* [ ] OCR.
* [ ] Layout detection support.
* [ ] Figure extraction support.
* [ ] Citation mapping support.
* [ ] Better handling of mixed document types.
* [ ] Compare retrieval quality across strategies.
* [ ] Revisit corpus-specific tuning.

---

## Re-indexing policy

* [ ] Any change to chunking logic must trigger a re-index plan.
* [ ] Existing collections indexed with old logic must be treated as stale.
* [ ] Define whether old collections are rebuilt, migrated, or retired.
* [ ] Keep the re-indexing decision explicit in the roadmap.

---

## Recommended execution order

1. Ingestion pipeline fixes
2. Chunk quality improvements
3. Parser upgrades
4. Retrieval power
5. Performance and cost
6. Reliability gates
7. Post E2E v1 upgrades
8. Re-indexing policy for old collections
