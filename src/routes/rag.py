from fastapi import APIRouter, Depends, UploadFile, File
from src.controllers.RAGController import RAGController
from src.models.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    StatsResponse,
)
from src.routes.dependencies import get_rag_controller

router = APIRouter(prefix="/api/v1", tags=["medical-rag"])


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    file: UploadFile = File(..., description="PDF, TXT, or Markdown medical document"),
    controller: RAGController = Depends(get_rag_controller),
):
    """
    Ingest a medical document (PDF, TXT, MD).

    PDF pipeline (automatic, no additional endpoints required):
    - Text extraction → Medical chunking → Embeddings
    - Table extraction (PyMuPDF) → Table summaries → Embeddings
    - Image extraction (PyMuPDF) → MedicalImageService → OCR → VLM Analysis
      → Findings extraction → Caption generation → Image metadata → Embeddings
    - All chunk types stored in unified Qdrant collection
    - Retrievable by the same hybrid search pipeline
    """
    return await controller.ingest(file)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    controller: RAGController = Depends(get_rag_controller),
):
    """
    Query the Medical Multimodal RAG system.

    Retrieves across text chunks, table chunks, and image-derived chunks
    using the same hybrid search pipeline.

    Full pipeline:
    Query → Query Enhancement → Embedding → Hybrid Retrieval (text+table+image) →
    Reranker (BGE+CrossEncoder+Medical) → Context Validation →
    Prompt Builder → LLM → Evidence Verification → Confidence Calibration

    Response: Answer + Sources (with chunk_type) + Confidence Score
    """
    return await controller.query(request)


@router.get("/health", response_model=HealthResponse)
async def health(controller: RAGController = Depends(get_rag_controller)):
    """System health check."""
    return await controller.health()


@router.get("/stats", response_model=StatsResponse)
async def stats(controller: RAGController = Depends(get_rag_controller)):
    """Collection statistics."""
    return await controller.stats()
