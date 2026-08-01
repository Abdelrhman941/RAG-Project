from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core import DistanceMetric
from ..retrieval import SearchResult
from ..schemas import PointData


class BaseVectorStore(ABC):
    """Interface every vector store implementation must satisfy.

    Sprint 7 introduced write-side operations (create / upsert / delete).
    Sprint 8 (Retrieval) extends this contract with `search`, the ONLY
    read-side operation. Adapters MUST NOT leak vendor-specific types
    across this boundary: `search` returns provider-agnostic
    `SearchResult` domain models.
    """

    # ---- Collection lifecycle ----
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Return True iff a collection with the given name exists."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: DistanceMetric,
    ) -> None:
        """Create a collection. Must be a no-op if it already exists."""

    # ---- Points ----
    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointData],
    ) -> int:
        """Insert or update the given points. Returns the number persisted."""

    @abstractmethod
    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        """Delete every point tagged with the given document_id.

        Used during re-indexing so the same document does not
        accumulate duplicate chunks across runs.
        """

    # ---- Retrieval ----
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return the top-k most similar points for the given query vector.

        Results are ordered by descending similarity as computed by the
        underlying engine — the caller does NOT re-rank. `min_score` is
        an optional lower bound: implementations MUST forward it to the
        engine (or filter server-side) so hits below the threshold are
        never returned. Adapters MUST map their native hit type into
        `SearchResult` before returning.
        """

    # ---- Health ----
    @abstractmethod
    async def health_check(self) -> bool:
        """Return True iff the backing store is reachable."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying client resources. Optional to override."""
        return None
