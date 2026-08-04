from uuid import uuid4

from app.retrieval.models import SearchResult
from app.services.retrieval_service import _deduplicate_parents


def test_deduplicate_parents_mixed_results() -> None:
    doc_id = uuid4()
    parent_x = uuid4()
    parent_y = uuid4()

    # 5 hits:
    # 0: Child A (Parent X) - score 0.9
    # 1: Flat chunk F (no parent) - score 0.8
    # 2: Child B (Parent X) - score 0.7
    # 3: Child C (Parent Y) - score 0.6
    # 4: Child D (Parent Y) - score 0.5

    hits = [
        SearchResult(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            score=0.9,
            content="Child A",
            parent_chunk_id=parent_x,
            parent_content="Parent X",
        ),
        SearchResult(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=1,
            page_number=1,
            score=0.8,
            content="Flat F",
        ),
        SearchResult(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=2,
            page_number=1,
            score=0.7,
            content="Child B",
            parent_chunk_id=parent_x,
            parent_content="Parent X",
        ),
        SearchResult(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=3,
            page_number=2,
            score=0.6,
            content="Child C",
            parent_chunk_id=parent_y,
            parent_content="Parent Y",
        ),
        SearchResult(
            document_id=doc_id,
            chunk_id=uuid4(),
            chunk_index=4,
            page_number=2,
            score=0.5,
            content="Child D",
            parent_chunk_id=parent_y,
            parent_content="Parent Y",
        ),
    ]

    results = _deduplicate_parents(hits)

    # Expecting 3 results:
    # 1. Parent X (taking score 0.9 from Child A)
    # 2. Flat F (score 0.8)
    # 3. Parent Y (taking score 0.6 from Child C)

    assert len(results) == 3

    # Check Parent X
    assert results[0].chunk_id == parent_x
    assert results[0].content == "Parent X"
    assert results[0].score == 0.9
    assert results[0].chunk_index == 0  # preserved from first child
    assert results[0].parent_chunk_id is None  # it is now the main content

    # Check Flat F
    assert results[1].content == "Flat F"
    assert results[1].score == 0.8
    assert results[1].parent_chunk_id is None

    # Check Parent Y
    assert results[2].chunk_id == parent_y
    assert results[2].content == "Parent Y"
    assert results[2].score == 0.6
    assert results[2].chunk_index == 3  # preserved from Child C

    # Verify we did not mutate original instances
    assert hits[0].content == "Child A"
