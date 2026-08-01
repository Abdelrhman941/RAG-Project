import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from app.core.enums.chunking import ChunkingStrategy
from app.core.exceptions import DocumentNotFoundError, InvalidChunkingParametersError
from app.services.document_chunker import chunk_document

pytestmark = pytest.mark.anyio


def _write_txt(upload_dir: Path, document_id: uuid.UUID, content: str) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / f"{document_id}.txt").write_text(content, encoding="utf-8")


async def test_chunk_document_assigns_sequential_chunk_index_across_pages(
    tmp_path: Path,
) -> None:
    document_id = uuid.uuid4()
    _write_txt(tmp_path, document_id, "a" * 1500)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=1000,
        overlap=200,
    )

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


async def test_chunk_document_stamps_correct_page_number(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    # TxtParser only ever produces a single page, so every chunk should be page 1.
    _write_txt(tmp_path, document_id, "content " * 200)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=300,
        overlap=50,
    )

    assert all(c.page_number == 1 for c in chunks)


async def test_chunk_document_stamps_shared_document_id(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    _write_txt(tmp_path, document_id, "content " * 200)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=300,
        overlap=50,
    )

    assert all(c.document_id == document_id for c in chunks)


async def test_chunk_document_assigns_unique_chunk_ids(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    _write_txt(tmp_path, document_id, "content " * 200)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=300,
        overlap=50,
    )

    chunk_ids = {c.chunk_id for c in chunks}
    assert len(chunk_ids) == len(chunks)


async def test_chunk_document_char_count_matches_content_length(
    tmp_path: Path,
) -> None:
    document_id = uuid.uuid4()
    _write_txt(tmp_path, document_id, "content " * 200)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=300,
        overlap=50,
    )

    assert all(c.char_count == len(c.content) for c in chunks)
    assert all(c.char_count == c.end_char - c.start_char for c in chunks)


async def test_chunk_document_raises_not_found_for_missing_document(
    tmp_path: Path,
) -> None:
    with pytest.raises(DocumentNotFoundError):
        await chunk_document(
            document_id=uuid.uuid4(),
            upload_dir=tmp_path,
            strategy=ChunkingStrategy.CHARACTER,
            chunk_size=300,
            overlap=50,
        )


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),  # chunk_size must be > 0
        (-100, 0),  # chunk_size must be > 0
        (300, -1),  # overlap must be >= 0
        (300, 300),  # overlap must be strictly smaller than chunk_size
        (300, 500),  # overlap larger than chunk_size
    ],
)
async def test_chunk_document_raises_for_invalid_parameters(
    tmp_path: Path, chunk_size: int, overlap: int
) -> None:
    document_id = uuid.uuid4()
    _write_txt(tmp_path, document_id, "content " * 200)

    with pytest.raises(InvalidChunkingParametersError):
        await chunk_document(
            document_id=document_id,
            upload_dir=tmp_path,
            strategy=ChunkingStrategy.CHARACTER,
            chunk_size=chunk_size,
            overlap=overlap,
        )


async def test_chunk_document_validates_parameters_before_touching_disk(
    tmp_path: Path,
) -> None:
    # No file was ever written for this document_id. If validation runs first,
    # we get InvalidChunkingParametersError, not DocumentNotFoundError.
    with pytest.raises(InvalidChunkingParametersError):
        await chunk_document(
            document_id=uuid.uuid4(),
            upload_dir=tmp_path,
            strategy=ChunkingStrategy.CHARACTER,
            chunk_size=0,
            overlap=0,
        )


async def test_chunk_document_with_pdf_preserves_page_numbers(
    tmp_path: Path, make_pdf_bytes: Callable[[list[str]], bytes]
) -> None:
    document_id = uuid.uuid4()
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf_bytes = make_pdf_bytes(["First page text", "Second page text"])
    (tmp_path / f"{document_id}.pdf").write_bytes(pdf_bytes)

    chunks = await chunk_document(
        document_id=document_id,
        upload_dir=tmp_path,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_size=1000,
        overlap=0,
    )

    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}
