from unittest.mock import AsyncMock, MagicMock

import pytest

from app.vectorstores.qdrant import QdrantVectorStore


@pytest.mark.asyncio
async def test_get_existing_hashes_returns_matched_hashes() -> None:
    store = QdrantVectorStore.__new__(QdrantVectorStore)

    # Two points already in store with content_hash payloads
    fake_point_1 = MagicMock()
    fake_point_1.payload = {"content_hash": "aabbcc"}
    fake_point_2 = MagicMock()
    fake_point_2.payload = {"content_hash": "ddeeff"}

    store._client = AsyncMock()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.scroll = AsyncMock(
        return_value=([fake_point_1, fake_point_2], None)
    )

    result = await store.get_existing_hashes(
        collection_name="test",
        hashes=frozenset({"aabbcc", "zzzzzz"}),
    )
    # Only "aabbcc" is in the store; "zzzzzz" is new
    assert result == frozenset({"aabbcc"})


@pytest.mark.asyncio
async def test_get_existing_hashes_empty_when_collection_missing() -> None:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = AsyncMock()
    store._client.collection_exists = AsyncMock(return_value=False)

    result = await store.get_existing_hashes(
        collection_name="missing",
        hashes=frozenset({"aabbcc"}),
    )
    assert result == frozenset()


@pytest.mark.asyncio
async def test_get_existing_hashes_empty_when_hashes_empty() -> None:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = AsyncMock()

    result = await store.get_existing_hashes(
        collection_name="test",
        hashes=frozenset(),
    )
    assert result == frozenset()
    store._client.collection_exists.assert_not_called()
