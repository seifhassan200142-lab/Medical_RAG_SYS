"""
Multimodal Retrieval Service
Retrieves across text, table, and image-derived chunks from the unified Qdrant collection.
Image chunks are stored in the same collection as text chunks (differentiated by chunk_type).
No separate image collection — all chunks are co-located and retrieved by the same pipeline.
"""
from typing import Optional

from src.models.ChunkModel import ChunkModel
from src.services.EmbeddingService import EmbeddingService
from src.stores.QdrantStore import QdrantStore
from src.helpers.logger import get_logger

logger = get_logger(__name__)


class MultimodalRetrievalService:
    """
    Provides combined retrieval that automatically includes text, table,
    and image-derived chunks. No separate image collection needed — all
    chunk types live in the main Qdrant collection and are retrieved together.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_store: QdrantStore,
    ):
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store

    def retrieve_combined(
        self,
        text_chunks: list[ChunkModel],
        query: str,
        document_id: Optional[str] = None,
    ) -> list[ChunkModel]:
        """
        text_chunks already contain image and table chunks since all chunk types
        are stored in the same collection. This method counts and logs them.
        """
        image_chunks = [c for c in text_chunks if c.metadata.get("is_image_chunk")]
        table_chunks = [c for c in text_chunks if c.metadata.get("chunk_type") == "table"]

        logger.info(
            f"Combined retrieval: {len(text_chunks)} total chunks | "
            f"image={len(image_chunks)} | table={len(table_chunks)}"
        )
        return text_chunks

    def count_chunk_types(self, chunks: list[ChunkModel]) -> dict[str, int]:
        return {
            "text": sum(1 for c in chunks if not c.metadata.get("is_image_chunk") and c.metadata.get("chunk_type") != "table"),
            "table": sum(1 for c in chunks if c.metadata.get("chunk_type") == "table"),
            "image": sum(1 for c in chunks if c.metadata.get("is_image_chunk")),
        }
