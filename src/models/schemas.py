from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Optional


# ── Request schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    top_k: Optional[int] = Field(None, ge=1, le=30)
    document_id: Optional[str] = None
    document_type: Optional[str] = None
    specialty: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    use_hyde: bool = True
    use_multi_query: bool = True
    use_parent_retrieval: bool = True
    use_compression: bool = True
    use_reranking: bool = True


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    citation_number: int
    chunk_id: str
    document_id: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    document_type: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    score: float
    rerank_score: Optional[float] = None
    text_preview: str
    chunk_type: str = "text"          # text | table | image
    is_image_source: bool = False
    image_modality: Optional[str] = None
    image_page: Optional[int] = None
    image_caption: Optional[str] = None


# ── Confidence ────────────────────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    overall: float
    retrieval_confidence: float
    answer_grounding: float
    source_coverage: float
    hallucination_risk: float
    is_reliable: bool
    warnings: list[str]


# ── Validation ────────────────────────────────────────────────────────────────

class ContextValidationInfo(BaseModel):
    is_valid: bool
    quality_score: float
    coverage_score: float
    consistency_score: float
    warnings: list[str]
    flagged_gaps: list[str]


class EvidenceVerificationInfo(BaseModel):
    support_ratio: float
    evidence_strength: str
    verified_claims_count: int
    unsupported_claims_count: int
    unsupported_warnings: list[str]


# ── Response schemas ──────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    confidence: ConfidenceBreakdown
    context_validation: Optional[ContextValidationInfo] = None
    evidence_verification: Optional[EvidenceVerificationInfo] = None
    retrieved_chunks: list[dict[str, Any]]
    query_variants: list[str]
    hyde_hypothesis: Optional[str] = None
    retrieval_strategy: str
    chunk_type_counts: dict[str, int] = {}
    reranking_applied: bool = False


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    chunks_created: int
    parent_chunks_created: int
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    images_extracted: int = 0
    tables_extracted: int = 0
    metadata: dict[str, Any]
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    qdrant_connected: bool
    embedding_model_loaded: bool
    reranker_available: bool
    collections: list[str]


class StatsResponse(BaseModel):
    collection_name: str
    total_vectors: int
    total_documents: int
    document_types: dict[str, int]
