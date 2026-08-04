from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core import DistanceMetric
from ..retrieval import SearchResult
from ..schemas import SparseVector
from .models import PointData


class BaseVectorStore(ABC):
    """Contract for all vector store implementations.

    Defines write (create/upsert/delete) and read (search) operations.
    Strictly enforces a provider-agnostic boundary: adapters must never leak
    vendor-specific types and must return domain models (e.g., `SearchResult`).
    """

    # ---- Collection lifecycle ----
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a specific collection already exists in the database."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: DistanceMetric,
    ) -> None:
        """Create a new collection with the required vector dimension & distance metric.
        Ignores the operation if the collection already exists."""

    # ---- Points ----
    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointData],
    ) -> int:
        """Insert new points or update existing ones in a collection.
        Returns the total number of successfully saved points."""

    @abstractmethod
    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        """Delete all vectors associated with a specific document ID.
        Crucial for re-indexing files without leaving duplicate old chunks behind."""

    @abstractmethod
    async def get_existing_hashes(
        self,
        collection_name: str,
        hashes: frozenset[str],
    ) -> frozenset[str]:
        """Return the subset of *hashes* already present in the collection.

        Used by the indexer to skip embedding and upserting chunks whose
        content has already been indexed. Implementations must query only
        the ``content_hash`` payload field and must not return vendor types.
        Returns an empty frozenset if the collection does not yet exist.
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
        sparse_vector: SparseVector | None = None,
    ) -> list[SearchResult]:
        """Find and return the `top_k` most similar vectors to the provided search vector.
        Optionally accepts a `sparse_vector` for hybrid search (if supported by the adapter).
        Optionally filters out results below the `min_score` threshold.
        Must return provider-agnostic `SearchResult` objects."""  # noqa: E501

    # ---- Health ----
    @abstractmethod
    async def health_check(self) -> bool:
        """Verify that the database connection is alive and responding."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Safely close the database connection and release resources."""
        return None
