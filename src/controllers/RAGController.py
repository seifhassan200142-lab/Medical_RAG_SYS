import os
import shutil
from datetime import datetime, timezone

from fastapi import UploadFile

from src.controllers.BaseController import BaseController
from src.services.IngestionService import IngestionService
from src.services.RetrievalService import RetrievalService
from src.services.GenerationService import GenerationService
from src.services.EmbeddingService import EmbeddingService
from src.services.RerankingService import RerankingService
from src.services.MultimodalRetrievalService import MultimodalRetrievalService
from src.stores.QdrantStore import QdrantStore
from src.stores.DocumentStore import DocumentStore
from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
    StatsResponse,
    ContextValidationInfo,
    EvidenceVerificationInfo,
)
from src.helpers.utils import sanitize_filename
from src.helpers.logger import get_logger
from src.helpers.config import settings

logger = get_logger(__name__)

UPLOAD_DIR = "uploads"


class RAGController(BaseController):
    def __init__(
        self,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        embedding_service: EmbeddingService,
        reranking_service: RerankingService,
        multimodal_service: MultimodalRetrievalService,
        qdrant_store: QdrantStore,
        document_store: DocumentStore,
    ):
        self.ingestion_service = ingestion_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
        self.multimodal_service = multimodal_service
        self.qdrant_store = qdrant_store
        self.document_store = document_store
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    async def ingest(self, file: UploadFile) -> IngestResponse:
        filename = sanitize_filename(file.filename or "upload")
        temp_path = os.path.join(UPLOAD_DIR, filename)
        try:
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()

        try:
            doc = self.ingestion_service.process_document(temp_path, filename)
            meta = doc.metadata or {}
            return IngestResponse(
                document_id=doc.id,
                filename=doc.filename,
                document_type=doc.document_type,
                chunks_created=doc.chunk_count,
                parent_chunks_created=doc.parent_chunk_count,
                text_chunks=meta.get("text_chunks", 0),
                table_chunks=meta.get("table_chunks", 0),
                image_chunks=meta.get("image_chunks", 0),
                images_extracted=meta.get("images_extracted", 0),
                tables_extracted=meta.get("tables_extracted", 0),
                metadata=meta,
                message=(
                    f"Successfully ingested '{filename}' as '{doc.document_type}'. "
                    f"Text chunks: {meta.get('text_chunks', 0)}, "
                    f"Table chunks: {meta.get('table_chunks', 0)}, "
                    f"Image chunks: {meta.get('image_chunks', 0)}."
                ),
            )
        except Exception as e:
            self.handle_exception(e)

    async def query(self, request: QueryRequest) -> QueryResponse:
        try:
            # Step 1: Hybrid Retrieval — returns text + table + image chunks together
            chunks, query_variants, hyde_hypothesis = self.retrieval_service.retrieve(
                question=request.question,
                top_k=request.top_k,
                document_id=request.document_id,
                document_type=request.document_type,
                specialty=request.specialty,
                year_from=request.year_from,
                year_to=request.year_to,
                use_hyde=request.use_hyde,
                use_multi_query=request.use_multi_query,
                use_parent_retrieval=request.use_parent_retrieval,
                use_compression=request.use_compression,
            )

            # Step 2: Count chunk types
            chunk_type_counts = self.multimodal_service.count_chunk_types(chunks)

            # Step 3: Reranking
            reranking_applied = False
            if request.use_reranking:
                top_n = request.top_k or settings.rerank_top_n
                chunks = self.reranking_service.rerank(
                    query=request.question,
                    chunks=chunks,
                    top_n=top_n,
                    use_bge=settings.enable_bge_reranker,
                    use_cross_encoder=settings.enable_cross_encoder,
                    use_medical=True,
                )
                reranking_applied = True

            # Step 4: Generation with validation and calibration
            answer, citations, confidence, ctx_validation, evidence_verification = (
                self.generation_service.generate(
                    question=request.question,
                    chunks=chunks,
                )
            )

            # Enrich citations with chunk-type metadata
            chunk_map = {c.chunk_id: c for c in chunks}
            for citation in citations:
                c = chunk_map.get(citation.chunk_id)
                if c:
                    citation.rerank_score = c.rerank_score
                    citation.chunk_type = c.metadata.get("chunk_type", "text")
                    citation.is_image_source = bool(c.metadata.get("is_image_chunk"))
                    citation.image_modality = c.metadata.get("modality")
                    citation.image_page = c.metadata.get("page")
                    citation.image_caption = c.metadata.get("caption")

            strategy_parts = [
                "Query Enhancement", "Embedding",
                "Hybrid Retrieval (dense+sparse, text+table+image)",
            ]
            if request.use_multi_query:
                strategy_parts.append(f"Multi-Query({len(query_variants)})")
            if request.use_hyde and hyde_hypothesis:
                strategy_parts.append("HyDE")
            if request.use_parent_retrieval:
                strategy_parts.append("Parent Retrieval")
            if request.use_compression:
                strategy_parts.append("Compression")
            if reranking_applied:
                strategy_parts.append("Reranker(BGE+CrossEncoder+Medical)")
            strategy_parts.extend(["Context Validation", "Prompt Builder", "LLM"])

            return QueryResponse(
                question=request.question,
                answer=answer,
                sources=citations,
                confidence=confidence,
                context_validation=ContextValidationInfo(
                    is_valid=ctx_validation.is_valid,
                    quality_score=ctx_validation.quality_score,
                    coverage_score=ctx_validation.coverage_score,
                    consistency_score=ctx_validation.consistency_score,
                    warnings=ctx_validation.warnings,
                    flagged_gaps=ctx_validation.flagged_gaps,
                ) if ctx_validation else None,
                evidence_verification=EvidenceVerificationInfo(
                    support_ratio=evidence_verification.support_ratio,
                    evidence_strength=evidence_verification.evidence_strength,
                    verified_claims_count=len(evidence_verification.verified_claims),
                    unsupported_claims_count=len(evidence_verification.unsupported_claims),
                    unsupported_warnings=[
                        c.warning for c in evidence_verification.unsupported_claims if c.warning
                    ],
                ) if evidence_verification else None,
                retrieved_chunks=[c.model_dump() for c in chunks],
                query_variants=query_variants,
                hyde_hypothesis=hyde_hypothesis,
                retrieval_strategy=" → ".join(strategy_parts),
                chunk_type_counts=chunk_type_counts,
                reranking_applied=reranking_applied,
            )
        except Exception as e:
            self.handle_exception(e)

    async def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc),
            qdrant_connected=self.qdrant_store.is_healthy(),
            embedding_model_loaded=self.embedding_service.is_loaded,
            reranker_available=(
                self.reranking_service.bge_reranker.is_available
                or self.reranking_service.cross_encoder.is_available
            ),
            collections=self.qdrant_store.list_collection_names(),
        )

    async def stats(self) -> StatsResponse:
        try:
            info = self.qdrant_store.get_collection_info()
        except Exception:
            info = {"total_vectors": 0}

        return StatsResponse(
            collection_name=self.qdrant_store.collection,
            total_vectors=info.get("total_vectors", 0),
            total_documents=self.document_store.count(),
            document_types=self.document_store.count_by_type(),
        )
