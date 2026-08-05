import asyncio
import logging
from uuid import uuid4

from app.core.config import get_settings
from app.core.enums import ChunkingStrategy
from app.embedders.factory import (
    get_embedding_provider,
    get_reranker,
    get_sparse_provider,
)
from app.services.document_indexer import index_document
from app.services.retrieval_service import retrieve
from app.vectorstores.factory import get_vector_store

logging.basicConfig(level=logging.WARNING)


async def test_small_to_big() -> None:
    settings = get_settings()
    doc_id = uuid4()

    # 1. Create a dummy large document (10 paragraphs)
    test_file_content = (
        "This is paragraph 1. It is short but has some unique words"
        " like ABRACADABRA.\n\n"
        "This is paragraph 2. Just filler text to make the document longer.\n\n"
        "This is paragraph 3. More filler text to stretch it out.\n\n"
        "This is paragraph 4. We want a document big enough to test"
        " hierarchical chunking.\n\n"
        "This is paragraph 5. The total length should exceed embedding"
        " chunk size.\n\n"
    ) * 4  # 20 paragraphs total

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = settings.UPLOAD_DIR / f"{doc_id}.txt"
    target_path.write_text(test_file_content)

    print("--- Testing Small-to-Big Retrieval ---")
    print(f"Total Document Length: {len(test_file_content)} chars")
    print(f"Embedding Chunk Size: {settings.DEFAULT_EMBEDDING_CHUNK_SIZE}")
    print(f"Prompt (Parent) Chunk Size: {settings.DEFAULT_PROMPT_CHUNK_SIZE}")

    # 2. Index the document
    dense_provider = get_embedding_provider()
    sparse_provider = get_sparse_provider()
    vector_store = get_vector_store()
    collection_name = f"test_small_big_{doc_id}"

    await index_document(
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

    # 3. Retrieve using the unique keyword in paragraph 1
    results = await retrieve(
        query="ABRACADABRA",
        collection_name=collection_name,
        provider=dense_provider,
        vector_store=vector_store,
        top_k=1,
        fetch_k=5,
        min_score=None,
        sparse_provider=sparse_provider,
        reranker=get_reranker(),
    )

    if results:
        top_result = results[0]
        print("\n--- Retrieval Success ---")
        print(
            f"Matched Child Chunk ID? (No, replaced with Parent): {top_result.chunk_id}"
        )
        print(f"Retrieved Content Length: {len(top_result.content)} chars")
        if len(top_result.content) > settings.DEFAULT_EMBEDDING_CHUNK_SIZE:
            print(
                "=> SUCCESS: The LLM receives the large PARENT context,"
                " not the small child chunk!"
            )
        else:
            print("=> FAILURE: The context is still small.")
    else:
        print("\n--- Retrieval Failed: No results found ---")

    # Cleanup
    await vector_store.delete_collection(collection_name)
    target_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_small_to_big())
