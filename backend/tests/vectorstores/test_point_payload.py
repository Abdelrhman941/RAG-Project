from uuid import uuid4

from app.core import SourceType
from app.vectorstores.models import PointPayload


def test_point_payload_carries_source_type_and_offsets() -> None:
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content="hello",
        source_type=SourceType.PDF,
        start_char=0,
        end_char=5,
    )
    assert payload.source_type == SourceType.PDF
    assert payload.start_char == 0
    assert payload.end_char == 5
