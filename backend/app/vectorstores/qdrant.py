import logging
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ..core import (
    DistanceMetric,
    IndexingError,
    VectorDimensionMismatchError,
    VectorStoreUnavailableError,
)
from ..schemas import PointData
from .base import BaseVectorStore

logger = logging.getLogger(__name__)

_DISTANCE_MAP: dict[DistanceMetric, qmodels.Distance] = {
    DistanceMetric.COSINE: qmodels.Distance.COSINE,
    DistanceMetric.DOT: qmodels.Distance.DOT,
    DistanceMetric.EUCLID: qmodels.Distance.EUCLID,
}


class QdrantVectorStore(BaseVectorStore):
    """Qdrant adapter.

    Owns the QdrantClient lifecycle and is the ONLY place in the codebase
    that imports `qdrant_client`. Any operation that would leak a Qdrant
    type across the interface (e.g. returning a `PointStruct`) is a bug.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: str | None = None,
    ) -> None:
        self._client = AsyncQdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
        )

    # ---- Collection lifecycle ----
    async def collection_exists(self, collection_name: str) -> bool:
        try:
            return await self._client.collection_exists(collection_name)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: DistanceMetric,
    ) -> None:
        if dimension <= 0:
            raise IndexingError("Embedding dimension must be greater than 0.")

        try:
            if await self._client.collection_exists(collection_name):
                await self._ensure_dimension_matches(collection_name, dimension)
                return

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=_DISTANCE_MAP[distance],
                ),
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc

    async def _ensure_dimension_matches(
        self, collection_name: str, dimension: int
    ) -> None:
        info = await self._client.get_collection(collection_name)
        params = info.config.params.vectors
        # `vectors` may be a single VectorParams or a dict of named vectors.
        # We only use the single-vector layout in this project.
        if isinstance(params, qmodels.VectorParams):
            existing = params.size
        elif isinstance(params, dict):
            existing = next(iter(params.values())).size
        else:  # pragma: no cover - defensive
            return
        if existing != dimension:
            raise VectorDimensionMismatchError(expected=existing, actual=dimension)

    # ---- Points ----
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointData],
    ) -> int:
        if not points:
            return 0

        structs = [
            qmodels.PointStruct(
                id=str(point.id),
                vector=point.vector,
                payload=point.payload.model_dump(mode="json"),
            )
            for point in points
        ]

        try:
            await self._client.upsert(
                collection_name=collection_name,
                points=structs,
                wait=True,
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise IndexingError(f"Failed to upsert points: {exc}") from exc

        return len(structs)

    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        selector = qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        )
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=selector,
                wait=True,
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            # A missing collection is fine on delete: nothing to remove.
            logger.warning("Delete-by-document skipped: %s", exc)

    # ---- Health ----
    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health check should never bubble
            return False

    async def close(self) -> None:
        await self._client.close()
