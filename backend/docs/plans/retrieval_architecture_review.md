# Retrieval Architecture Audit: Hierarchical Chunking (Small-to-Big)

To achieve "small-to-big" retrieval, the system must match on highly granular "child" chunks but provide the LLM with the broader "parent" context. Because our current architecture relies exclusively on a single Vector Database (Qdrant) without a secondary Document/SQL store, the retrieval architecture must be chosen carefully.

Below is an evaluation of three potential designs.

---

## Design 1: Parent Content Duplicated in Every Child Payload
**Concept:** The chunker performs two passes. Each emitted child chunk carries its own embedding, but its `PointPayload` contains both the `parent_chunk_id` and the full raw string of the `parent_content`.
* **Storage Cost:** **High**. If a 1000-token parent is split into five 200-token children, the parent string is duplicated 5 times.
* **Retrieval Latency:** **Very Low (O(1))**. A single similarity search returns the top children. The retriever extracts the `parent_content` from the payloads, deduplicates them in-memory by `parent_chunk_id`, and feeds the LLM.
* **Implementation Complexity:** **Low**. Requires zero changes to the vector store interface, no new database tables, and no multi-step retrieval logic.
* **Qdrant Compatibility:** Perfect. Qdrant handles arbitrary JSON payloads effortlessly.
* **Scalability:** Good for compute/latency, but memory/disk usage on the vector DB grows linearly with the duplication factor. Text is generally cheap compared to HNSW vector indices, though.
* **Production Usage:** Standard practice for lightweight RAG applications and the default fallback in frameworks like LlamaIndex when a dedicated document store is unavailable.

---

## Design 2: Parent ID Reference with Parent Lookup (KV Store)
**Concept:** Child chunks store only a `parent_chunk_id` in their payload. The actual parent chunks are stored in a secondary Key-Value (KV) store (e.g., Postgres, Redis, MongoDB) or alongside the children as "dummy" points.
* **Storage Cost:** **Low**. The parent text is stored exactly once.
* **Retrieval Latency:** **Moderate (O(2))**. Requires a similarity search for children, followed by a batch-get by ID from the KV store.
* **Implementation Complexity:** **Moderate to High**. We currently lack a Document Store. If we force Qdrant to act as the KV store, we must inject parent points into the same collection (requiring dummy vectors, as our `PointData` schema currently mandates a float array).
* **Qdrant Compatibility:** Requires architectural workarounds (dummy vectors) or upgrading to Qdrant's named vectors to allow vector-less points.
* **Scalability:** Excellent. This is the most storage-efficient model.
* **Production Usage:** The gold standard for massive enterprise RAG systems with millions of documents.

---

## Design 3: Separate Parent/Child Collections
**Concept:** Collection A holds embedded child chunks. Collection B holds embedded (or purely payload) parent chunks.
* **Storage Cost:** **Moderate**. No duplication, but requires maintaining two separate Qdrant collections.
* **Retrieval Latency:** **Moderate (O(2))**. Search Collection A -> Batch retrieve IDs from Collection B.
* **Implementation Complexity:** **High**. Breaks our current `BaseVectorStore` contract, which assumes a 1:1 mapping between an index and a collection. Requires distributed transaction logic to ensure dual-upserts don't tear.
* **Qdrant Compatibility:** Fully compatible natively, but burdens the application layer.
* **Scalability:** Overkill. Doubles the operational overhead (snapshots, backups, RAM overhead for collection metadata).
* **Production Usage:** Typically only used if the parent chunks *also* need to be searchable independently (e.g., semantic search over summaries vs. semantic search over micro-chunks).

---

## Recommendation: Design 1 (Payload Duplication)

I recommend **Design 1: Parent Content Duplicated in Every Child Payload** for this project.

### Why?
1. **No Infrastructure Creep:** We do not currently have a relational database or KV store. Forcing Qdrant to act as one (Design 2) requires architectural hacks (dummy vectors) that pollute the clean `BaseVectorStore` abstraction we just built.
2. **Minimal Complexity:** We keep the pipeline purely stateless. The `chunk_document` function emits children enriched with parent strings, and the vector store blindly upserts them. 
3. **Latency over Storage:** In modern RAG, vector index RAM (HNSW graphs) is the primary cost driver, not raw string storage. A 768-d vector uses ~3KB of RAM. Duplicating 1000 tokens of text (~4KB) on disk in Qdrant's payload is a negligible trade-off to achieve single-hop, lightning-fast retrievals.
4. **Agility:** This can be implemented in a single step (Step 11) without rewriting the entire indexer orchestration. If storage costs become an issue later at scale, we can easily migrate to Design 2 by spinning up a Redis/Postgres instance and stripping the payloads.
