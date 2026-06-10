from langchain_huggingface import HuggingFaceEmbeddings
from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.helpers.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Dense embedding service using BGE-M3 (1024-dim).
    BGE-M3 supports dense, sparse, and ColBERT retrieval from a single model.
    We use the HuggingFace wrapper for dense embeddings.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            self._model = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
            logger.info("Embedding model loaded")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._model.embed_query(text)
        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            logger.info(f"Embedding {len(texts)} texts")
            return self._model.embed_documents(texts)
        except Exception as e:
            raise EmbeddingError(f"Document embedding failed: {e}") from e

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
