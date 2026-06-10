from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    SearchRequest,
    HasIdCondition,
    PointIdsList,
)
from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.helpers.exceptions import StorageError

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection = settings.collection_name
        self.parent_collection = settings.parent_collection_name

    def create_collections(self, vector_size: int = 1024) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Created hybrid collection '{self.collection}'")

        if self.parent_collection not in existing:
            self.client.create_collection(
                collection_name=self.parent_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created parent collection '{self.parent_collection}'")

    def upsert_chunks(self, points: list[PointStruct]) -> None:
        try:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            logger.info(f"Upserted {len(points)} child chunks")
        except Exception as e:
            raise StorageError(f"Failed to upsert chunks: {e}") from e

    def upsert_parents(self, points: list[PointStruct]) -> None:
        try:
            self.client.upsert(collection_name=self.parent_collection, points=points, wait=True)
            logger.info(f"Upserted {len(points)} parent chunks")
        except Exception as e:
            raise StorageError(f"Failed to upsert parent chunks: {e}") from e

    def dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        score_threshold: float = 0.0,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        try:
            return self.client.search(
                collection_name=self.collection,
                query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=query_vector),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as e:
            raise StorageError(f"Dense search failed: {e}") from e

    def sparse_search(
        self,
        indices: list[int],
        values: list[float],
        top_k: int,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        try:
            return self.client.search(
                collection_name=self.collection,
                query_vector=NamedSparseVector(
                    name=SPARSE_VECTOR_NAME,
                    vector=SparseVector(indices=indices, values=values),
                ),
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as e:
            raise StorageError(f"Sparse search failed: {e}") from e

    def get_parents_by_ids(self, parent_ids: list[str]) -> list[ScoredPoint]:
        try:
            results = self.client.retrieve(
                collection_name=self.parent_collection,
                ids=parent_ids,
                with_payload=True,
                with_vectors=False,
            )
            return results
        except Exception as e:
            raise StorageError(f"Parent retrieval failed: {e}") from e

    def build_filter(
        self,
        document_id: str | None = None,
        document_type: str | None = None,
        specialty: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> Filter | None:
        conditions = []

        if document_id:
            conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )
        if document_type:
            conditions.append(
                FieldCondition(key="document_type", match=MatchValue(value=document_type))
            )
        if specialty:
            conditions.append(
                FieldCondition(key="specialty", match=MatchValue(value=specialty))
            )
        if year_from or year_to:
            conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=year_from if year_from else None,
                        lte=year_to if year_to else None,
                    ),
                )
            )

        return Filter(must=conditions) if conditions else None

    def delete_by_document_id(self, document_id: str) -> None:
        try:
            f = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
            self.client.delete(collection_name=self.collection, points_selector=f)
            self.client.delete(collection_name=self.parent_collection, points_selector=f)
            logger.info(f"Deleted all chunks for document_id={document_id}")
        except Exception as e:
            raise StorageError(f"Failed to delete document: {e}") from e

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "name": self.collection,
            "total_vectors": info.points_count,
            "status": str(info.status),
        }

    def list_collection_names(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
