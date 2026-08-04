import hashlib
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core import SourceType
from app.vectorstores.models import PointPayload


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_point_payload_includes_content_hash() -> None:
    text = "Some chunk content."
    payload = PointPayload(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        page_number=1,
        content=text,
        source_type=SourceType.TXT,
        start_char=0,
        end_char=len(text),
        content_hash=_sha256(text),
    )
    assert payload.content_hash == _sha256(text)


def test_point_payload_rejects_missing_content_hash() -> None:
    with pytest.raises(ValidationError):
        PointPayload(  # type: ignore[call-arg]
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=1,
            content="text",
            source_type=SourceType.TXT,
            start_char=0,
            end_char=4,
        )
