import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "medical_rag"
    parent_collection_name: str = "medical_rag_parents"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Sparse / BM25
    bm25_collection_name: str = "medical_rag_bm25"

    # Retrieval
    top_k: int = 8
    parent_top_k: int = 3
    score_threshold: float = 0.3
    hybrid_alpha: float = 0.6
    multi_query_count: int = 3
    context_compression_ratio: float = 0.6
    rerank_top_n: int = 5

    # Reranking
    bge_reranker_model: str = "BAAI/bge-reranker-base"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_bge_reranker: bool = True
    enable_cross_encoder: bool = True

    # Image processing (PDF-embedded images)
    image_store_path: str = "uploads/images"
    min_image_width: int = 80
    min_image_height: int = 80
    min_image_bytes: int = 4096
    max_image_dim: int = 1024

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 2048
    parent_chunk_overlap: int = 128

    # Medical NLP
    medical_section_headers: list = field(default_factory=lambda: [
        "abstract", "introduction", "background", "methods", "methodology",
        "results", "discussion", "conclusion", "references", "clinical",
        "diagnosis", "treatment", "findings", "history", "examination",
        "assessment", "plan", "medications", "allergies", "labs",
        "imaging", "pathology", "prognosis", "recommendations",
    ])

    # HyDE
    hyde_temperature: float = 0.3
    hyde_max_tokens: int = 256

    # Confidence
    confidence_threshold: float = 0.55
    hallucination_threshold: float = 0.4

    # App
    app_env: str = "development"
    log_level: str = "INFO"


def load_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    return Settings(
        groq_api_key=groq_api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv("COLLECTION_NAME", "medical_rag"),
        parent_collection_name=os.getenv("PARENT_COLLECTION_NAME", "medical_rag_parents"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        top_k=int(os.getenv("TOP_K", "8")),
        parent_top_k=int(os.getenv("PARENT_TOP_K", "3")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.3")),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.6")),
        multi_query_count=int(os.getenv("MULTI_QUERY_COUNT", "3")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        bge_reranker_model=os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base"),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        enable_bge_reranker=os.getenv("ENABLE_BGE_RERANKER", "true").lower() == "true",
        enable_cross_encoder=os.getenv("ENABLE_CROSS_ENCODER", "true").lower() == "true",
        image_store_path=os.getenv("IMAGE_STORE_PATH", "uploads/images"),
        min_image_width=int(os.getenv("MIN_IMAGE_WIDTH", "80")),
        min_image_height=int(os.getenv("MIN_IMAGE_HEIGHT", "80")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", "2048")),
        parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", "128")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.55")),
        hallucination_threshold=float(os.getenv("HALLUCINATION_THRESHOLD", "0.4")),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = load_settings()
