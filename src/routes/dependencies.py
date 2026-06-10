from functools import lru_cache

from src.services.EmbeddingService import EmbeddingService
from src.services.IngestionService import IngestionService
from src.services.RetrievalService import RetrievalService
from src.services.GenerationService import GenerationService
from src.services.QueryEnhancementService import QueryEnhancementService
from src.services.RerankingService import RerankingService
from src.services.MultimodalRetrievalService import MultimodalRetrievalService
from src.services.MedicalImageService import MedicalImageService
from src.stores.QdrantStore import QdrantStore
from src.stores.DocumentStore import DocumentStore
from src.medical.bm25_encoder import BM25Encoder
from src.controllers.RAGController import RAGController
from src.helpers.config import settings


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    store = QdrantStore()
    store.create_collections(vector_size=settings.embedding_dim)
    return store


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    return DocumentStore()


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    return GenerationService()


@lru_cache(maxsize=1)
def get_query_enhancement_service() -> QueryEnhancementService:
    return QueryEnhancementService()


@lru_cache(maxsize=1)
def get_bm25_encoder() -> BM25Encoder:
    return BM25Encoder()


@lru_cache(maxsize=1)
def get_reranking_service() -> RerankingService:
    return RerankingService(
        bge_model=settings.bge_reranker_model,
        cross_encoder_model=settings.cross_encoder_model,
    )


@lru_cache(maxsize=1)
def get_medical_image_service() -> MedicalImageService:
    return MedicalImageService(image_store_path=settings.image_store_path)


@lru_cache(maxsize=1)
def get_multimodal_service() -> MultimodalRetrievalService:
    return MultimodalRetrievalService(
        embedding_service=get_embedding_service(),
        qdrant_store=get_qdrant_store(),
    )


def get_rag_controller() -> RAGController:
    embedding_service = get_embedding_service()
    qdrant_store = get_qdrant_store()
    document_store = get_document_store()
    generation_service = get_generation_service()
    query_enhancement = get_query_enhancement_service()
    bm25_encoder = get_bm25_encoder()
    reranking_service = get_reranking_service()
    multimodal_service = get_multimodal_service()
    image_service = get_medical_image_service()

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        qdrant_store=qdrant_store,
        document_store=document_store,
        bm25_encoder=bm25_encoder,
        image_service=image_service,
    )
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        qdrant_store=qdrant_store,
        query_enhancement=query_enhancement,
        bm25_encoder=bm25_encoder,
    )

    return RAGController(
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        embedding_service=embedding_service,
        reranking_service=reranking_service,
        multimodal_service=multimodal_service,
        qdrant_store=qdrant_store,
        document_store=document_store,
    )
