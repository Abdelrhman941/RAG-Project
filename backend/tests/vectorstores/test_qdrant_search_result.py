from uuid import uuid4

from qdrant_client.http import models as qmodels

from app.vectorstores.qdrant import QdrantVectorStore


def test_to_search_result_extracts_parent_fields() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    parent_id = uuid4()

    point = qmodels.ScoredPoint(
        id=str(chunk_id),
        version=1,
        score=0.95,
        payload={
            "document_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": "child chunk",
            "parent_chunk_id": str(parent_id),
            "parent_content": "parent chunk",
        },
        vector=None,
    )

    result = QdrantVectorStore._to_search_result(point)
    assert result.parent_chunk_id == parent_id
    assert result.parent_content == "parent chunk"
    assert result.content == "child chunk"
    assert result.chunk_id == chunk_id


def test_to_search_result_handles_missing_parent_fields() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()

    point = qmodels.ScoredPoint(
        id=str(chunk_id),
        version=1,
        score=0.95,
        payload={
            "document_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": "child chunk",
        },
        vector=None,
    )

    result = QdrantVectorStore._to_search_result(point)
    assert result.parent_chunk_id is None
    assert result.parent_content is None
