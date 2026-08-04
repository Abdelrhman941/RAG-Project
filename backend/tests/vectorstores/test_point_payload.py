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
