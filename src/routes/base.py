from fastapi import APIRouter, Depends
from src.controllers.RAGController import RAGController
from src.models.schemas import HealthResponse, StatsResponse
from src.routes.dependencies import get_rag_controller

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(controller: RAGController = Depends(get_rag_controller)):
    return await controller.health()


@router.get("/stats", response_model=StatsResponse)
async def stats(controller: RAGController = Depends(get_rag_controller)):
    return await controller.stats()


@router.get("/")
async def root():
    return {
        "name": "Medical Multimodal RAG API",
        "version": "2.0.0",
        "docs": "/docs",
    }
