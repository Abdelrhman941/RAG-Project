from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core.enums.vector_store import DistanceMetric
from ..schemas import PointData


class BaseVectorStore(ABC):
    """Interface every vector store implementation must satisfy.

    Kept intentionally narrow for Sprint 7: this sprint only stores data.
    Similarity search lives in Sprint 8 (Retrieval) and will be
    added as a separate method (`search`) later.
    Adapters MUST NOT leak vendor-specific types across this boundary.
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

    # ---- Health ----
    @abstractmethod
    async def health_check(self) -> bool:
        """Return True iff the backing store is reachable."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying client resources. Optional to override."""
        return None
