import hashlib
from uuid import uuid4

from app.core import SourceType
from app.vectorstores.models import PointPayload


def test_point_payload_carries_source_type_and_offsets() -> None:
    text = "hello"
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content=text,
        source_type=SourceType.PDF,
        start_char=0,
        end_char=5,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    assert payload.source_type == SourceType.PDF
    assert payload.start_char == 0
    assert payload.end_char == 5


def test_point_payload_accepts_parent_fields() -> None:
    doc_id = uuid4()
    parent_id = uuid4()

    # Test without parent fields
    payload_no_parent = PointPayload(
        document_id=doc_id,
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="test",
        source_type=SourceType.PDF,
        start_char=0,
        end_char=100,
        content_hash="abc",
        parent_chunk_id=None,
        parent_content=None,
    )
    assert payload_no_parent.document_id == doc_id
    assert payload_no_parent.parent_chunk_id is None
    assert payload_no_parent.parent_content is None

    # Test with parent fields
    payload = PointPayload(
        document_id=doc_id,
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="test",
        source_type=SourceType.PDF,
        start_char=0,
        end_char=4,
        content_hash="abc",
        parent_chunk_id=parent_id,
        parent_content="test parent",
    )
    assert payload.parent_chunk_id == parent_id
    assert payload.parent_content == "test parent"


def test_point_payload_accepts_payload_with_source_type_and_offsets() -> None:
    text = "test chunk"
    h = hashlib.sha256(text.encode()).hexdigest()
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=1,
        page_number=2,
        content=text,
        source_type=SourceType.TXT,
        start_char=0,
        end_char=len(text),
        content_hash=h,
    )
    assert payload.content_hash == h
