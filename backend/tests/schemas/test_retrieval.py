from uuid import uuid4

from app.retrieval.models import SearchResult


def test_search_result_accepts_parent_fields() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    parent_id = uuid4()

    result = SearchResult(
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        page_number=1,
        score=0.95,
        content="child content",
        parent_chunk_id=parent_id,
        parent_content="parent content",
    )

    assert result.parent_chunk_id == parent_id
    assert result.parent_content == "parent content"


def test_search_result_defaults_to_none_for_parent() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()

    result = SearchResult(
        document_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=0,
        page_number=1,
        score=0.95,
        content="child content",
    )

    assert result.parent_chunk_id is None
    assert result.parent_content is None
