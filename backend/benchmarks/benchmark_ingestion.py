import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.core.enums import ChunkingStrategy
from app.embedders.factory import (
    get_embedding_provider,
    get_sparse_provider,
)
from app.services.document_chunker import chunk_document
from app.services.document_indexer import index_document
from app.services.document_parser import parse_document
from app.vectorstores.factory import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).parent
CORPUS_DIR = BENCHMARKS_DIR / "corpus"
RESULTS_DIR = BENCHMARKS_DIR / "results"


async def benchmark_document(file_path: Path) -> dict[str, Any]:
    settings = get_settings()
    doc_id = uuid4()

    # We copy the file to the upload_dir for the services to find it
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = settings.UPLOAD_DIR / f"{doc_id}{file_path.suffix}"
    target_path.write_bytes(file_path.read_bytes())

    # The document_name parameter isn't currently used heavily by chunker,
    # but parse_document uses upload_dir and document_id
    # We need to simulate the upload by putting the file in the right place

    metrics = {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
    }

    # 1. Parse
    start_time = time.perf_counter()
    pages_gen, source_type = await parse_document(
        document_id=doc_id,
        upload_dir=settings.UPLOAD_DIR,
    )
    pages = [p async for p in pages_gen]
    parse_time = time.perf_counter() - start_time
    metrics["parse_time_sec"] = parse_time
    metrics["parsed_length_chars"] = sum(len(p) for p in pages)

    # 2. Chunk
    start_time = time.perf_counter()
    chunks_gen = chunk_document(
        document_id=doc_id,
        upload_dir=settings.UPLOAD_DIR,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=settings.DEFAULT_EMBEDDING_CHUNK_SIZE,
        embedding_overlap=settings.DEFAULT_EMBEDDING_OVERLAP,
        prompt_chunk_size=settings.DEFAULT_PROMPT_CHUNK_SIZE,
        prompt_overlap=settings.DEFAULT_PROMPT_OVERLAP,
    )
    chunks = [c async for c in chunks_gen]
    chunk_time = time.perf_counter() - start_time
    metrics["chunk_time_sec"] = chunk_time
    metrics["num_chunks"] = len(chunks)
    if chunks:
        metrics["avg_chunk_chars"] = sum(c.char_count for c in chunks) / len(chunks)
    else:
        metrics["avg_chunk_chars"] = 0

    # 3. Index (includes embedding and upsert)
    dense_provider = get_embedding_provider()
    sparse_provider = get_sparse_provider()
    vector_store = get_vector_store()

    # We need a dedicated collection for benchmark to avoid polluting real data
    collection_name = f"benchmark_{doc_id}"

    start_time = time.perf_counter()
    result = await index_document(
        document_id=doc_id,
        upload_dir=settings.UPLOAD_DIR,
        collection_name=collection_name,
        distance=settings.DISTANCE_METRIC,
        provider=dense_provider,
        vector_store=vector_store,
        strategy=ChunkingStrategy.TOKEN,
        embedding_chunk_size=settings.DEFAULT_EMBEDDING_CHUNK_SIZE,
        embedding_overlap=settings.DEFAULT_EMBEDDING_OVERLAP,
        prompt_chunk_size=settings.DEFAULT_PROMPT_CHUNK_SIZE,
        prompt_overlap=settings.DEFAULT_PROMPT_OVERLAP,
        sparse_provider=sparse_provider,
    )
    index_time = time.perf_counter() - start_time
    metrics["index_time_sec"] = index_time
    metrics["indexed_points"] = result.indexed_points

    # Clean up collection
    try:
        await vector_store.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Could not delete benchmark collection: {e}")

    return metrics


async def run_benchmarks() -> None:
    if not CORPUS_DIR.exists():
        logger.error(
            f"Corpus directory {CORPUS_DIR} not found. Run generate_corpus.py first."
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = list(CORPUS_DIR.glob("*.*"))
    if not files:
        logger.error("No files found in corpus directory.")
        return

    logger.info(f"Starting benchmarks on {len(files)} files...")
    results = []
    for f in files:
        logger.info(f"Benchmarking {f.name}...")
        try:
            metrics = await benchmark_document(f)
            results.append(metrics)
            logger.info(f"Finished {f.name}: {metrics}")
        except Exception as e:
            logger.error(f"Failed to benchmark {f.name}: {e}")

    output_path = RESULTS_DIR / "baseline.json"
    with output_path.open("w") as out:
        json.dump(results, out, indent=2)

    logger.info(f"Saved benchmark results to {output_path}")

    # Print summary
    print("\n--- Benchmark Summary ---")
    for r in results:
        print(f"File: {r.get('file_name', 'Unknown')}")
        print(f"  Size: {r.get('file_size_bytes', 0) / 1024:.2f} KB")
        print(f"  Parse: {r.get('parse_time_sec', 0):.4f}s")
        print(f"  Chunk: {r.get('chunk_time_sec', 0):.4f}s")
        print(f"  Index: {r.get('index_time_sec', 0):.4f}s")
        print(f"  Chunks: {r.get('num_chunks', 0)}")
        print(f"  Avg Chunk Size: {r.get('avg_chunk_chars', 0):.0f} chars")
        print("-" * 25)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
