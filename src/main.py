from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from src.routes import base, rag
from src.routes.dependencies import get_embedding_service, get_qdrant_store
from src.helpers.logger import get_logger
from src.helpers.exceptions import RAGException

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical RAG system...")
    get_embedding_service()
    get_qdrant_store()
    logger.info("Medical RAG system ready")
    yield
    logger.info("Shutting down Medical RAG system")


app = FastAPI(
    title="Medical Multimodal RAG API",
    description=(
        "Production-grade Medical RAG system with Hybrid Search (BM25 + Dense), "
        "Multi-Query Retrieval, HyDE, Parent Document Retrieval, "
        "Context Compression, Citation Generation, and Confidence Scoring. "
        "Powered by BGE-M3, Qdrant, and Groq LLaMA 3."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(base.router)
app.include_router(rag.router)
