import logging
from collections.abc import Sequence
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ..core import (
    CollectionNotFoundError,
    DistanceMetric,
    IndexingError,
    RetrievalError,
    VectorDimensionMismatchError,
    VectorStoreUnavailableError,
)
from ..retrieval import SearchResult
from .base import BaseVectorStore
from .models import PointData

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
    type across the interface (e.g. returning a `PointStruct` or a
    `ScoredPoint`) is a bug.
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
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                return
            raise VectorStoreUnavailableError(
                f"Failed to delete points for document '{document_id}': {exc}"
            ) from exc
        except ResponseHandlingException as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant is unreachable while deleting document '{document_id}': {exc}"
            ) from exc

    async def get_existing_hashes(
        self,
        collection_name: str,
        hashes: frozenset[str],
    ) -> frozenset[str]:
        """Return the subset of *hashes* already present in the collection.

        Uses a payload-filter scroll with ``MatchAny`` so only matching
        points are fetched — network overhead stays proportional to the
        number of candidate chunks, not the full collection size.
        Returns an empty frozenset immediately when *hashes* is empty or
        the collection does not yet exist.
        """
        if not hashes:
            return frozenset()
        try:
            exists = await self._client.collection_exists(collection_name)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc
        if not exists:
            return frozenset()

        try:
            points, _ = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="content_hash",
                            match=qmodels.MatchAny(any=list(hashes)),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=len(hashes),
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant hash lookup failed: {exc}"
            ) from exc

        found: set[str] = set()
        for point in points:
            if point.payload and "content_hash" in point.payload:
                h = str(point.payload["content_hash"])
                if h in hashes:
                    found.add(h)
        return frozenset(found)

    # ---- Retrieval ----
    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Run a top-k semantic-similarity query against Qdrant.

        Uses `query_points` (the modern Qdrant search API) and maps
        each returned `ScoredPoint` into a domain-level `SearchResult`.
        Ranking is done entirely by Qdrant; we never re-sort here.
        """
        if top_k <= 0:
            return []
        try:
            response = await self._client.query_points(
                collection_name=collection_name,
                query=list(vector),
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            # Qdrant returns 404 when the collection is missing.
            if getattr(exc, "status_code", None) == 404:
                raise CollectionNotFoundError(collection_name) from exc
            raise RetrievalError(f"Qdrant search failed: {exc}") from exc
        except ResponseHandlingException as exc:
            raise VectorStoreUnavailableError(f"Qdrant is unreachable: {exc}") from exc
        return [self._to_search_result(point) for point in response.points]

    @staticmethod
    def _to_search_result(point: qmodels.ScoredPoint) -> SearchResult:
        """Map a Qdrant `ScoredPoint` into the provider-agnostic domain model.

        Kept as a small static helper so the mapping is trivially unit-
        testable without spinning up an AsyncQdrantClient.
        """
        payload = point.payload or {}
        try:
            return SearchResult(
                document_id=UUID(str(payload["document_id"])),
                chunk_id=UUID(str(payload["chunk_id"])),
                chunk_index=int(payload["chunk_index"]),
                page_number=int(payload["page_number"]),
                score=float(point.score),
                content=str(payload["content"]),
                parent_chunk_id=UUID(str(payload["parent_chunk_id"]))
                if payload.get("parent_chunk_id")
                else None,
                parent_content=str(payload["parent_content"])
                if payload.get("parent_content")
                else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Malformed point payload for id={point.id}: {exc}"
            ) from exc

    # ---- Health ----
    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health check should never bubble
            return False

    async def close(self) -> None:
        await self._client.close()
