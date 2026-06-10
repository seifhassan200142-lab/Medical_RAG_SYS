"""
Reranking Service
Implements:
  - BGE Reranker (FlagEmbedding)
  - Cross-Encoder Reranker (sentence-transformers)
  - Medical Reranking Pipeline combining both with domain-specific scoring
"""
import re
from typing import Optional

from src.models.ChunkModel import ChunkModel
from src.helpers.config import settings
from src.helpers.logger import get_logger

logger = get_logger(__name__)

MEDICAL_PRIORITY_TERMS = [
    "clinical trial", "randomized controlled", "meta-analysis", "systematic review",
    "evidence-based", "guidelines", "recommendation", "contraindicated",
    "drug interaction", "adverse effect", "side effect", "mechanism of action",
    "pathophysiology", "etiology", "prognosis", "mortality", "morbidity",
    "sensitivity", "specificity", "positive predictive", "negative predictive",
    "first-line", "second-line", "standard of care", "treatment protocol",
]

MEDICAL_NEGATIVE_TERMS = [
    "disclaimer", "not medical advice", "consult your doctor",
    "this information is for educational", "general information only",
]


class BGEReranker:
    """
    BGE Reranker using FlagEmbedding's FlagReranker.
    Falls back gracefully if model unavailable.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)
            logger.info(f"BGE Reranker loaded: {self.model_name}")
        except ImportError:
            logger.warning("FlagEmbedding not installed — BGE reranker disabled. Install with: pip install FlagEmbedding")
        except Exception as e:
            logger.warning(f"BGE Reranker load failed: {e} — falling back to score passthrough")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, chunks: list[ChunkModel], top_n: Optional[int] = None) -> list[ChunkModel]:
        if not self._model or not chunks:
            return chunks

        try:
            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self._model.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]

            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = round(float(score), 6)

            reranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
            return reranked[:top_n] if top_n else reranked
        except Exception as e:
            logger.warning(f"BGE reranking failed: {e} — returning original order")
            return chunks


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker using sentence-transformers.
    Uses ms-marco-MiniLM model optimized for passage ranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Cross-Encoder Reranker loaded: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed — cross-encoder reranker disabled.")
        except Exception as e:
            logger.warning(f"Cross-Encoder load failed: {e} — falling back to score passthrough")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, chunks: list[ChunkModel], top_n: Optional[int] = None) -> list[ChunkModel]:
        if not self._model or not chunks:
            return chunks

        try:
            import numpy as np
            pairs = [(query, chunk.text) for chunk in chunks]
            raw_scores = self._model.predict(pairs)

            def sigmoid(x):
                return float(1.0 / (1.0 + np.exp(-x)))

            for chunk, raw_score in zip(chunks, raw_scores):
                normalized = sigmoid(float(raw_score))
                chunk.rerank_score = round(normalized, 6)

            reranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
            return reranked[:top_n] if top_n else reranked
        except Exception as e:
            logger.warning(f"Cross-Encoder reranking failed: {e} — returning original order")
            return chunks


class MedicalReranker:
    """
    Medical domain-specific re-scoring applied on top of base rerankers.
    Boosts clinically relevant chunks, penalizes low-quality content.
    """

    def __init__(self):
        self._medical_terms_set = {t.lower() for t in MEDICAL_PRIORITY_TERMS}
        self._negative_terms_set = {t.lower() for t in MEDICAL_NEGATIVE_TERMS}

    def compute_medical_score(self, chunk: ChunkModel, query: str) -> float:
        text_lower = chunk.text.lower()
        query_lower = query.lower()

        priority_hits = sum(1 for t in self._medical_terms_set if t in text_lower)
        negative_hits = sum(1 for t in self._negative_terms_set if t in text_lower)

        medical_boost = min(priority_hits * 0.03, 0.15)
        negative_penalty = negative_hits * 0.05

        section_boost = 0.0
        high_value_sections = {"results", "conclusion", "findings", "assessment", "methods"}
        if chunk.section and chunk.section.lower() in high_value_sections:
            section_boost = 0.05

        doc_type_boost = 0.0
        high_value_types = {"research_paper", "medical_guideline", "clinical_note"}
        if chunk.document_type and chunk.document_type in high_value_types:
            doc_type_boost = 0.05

        length_score = min(len(chunk.text) / 1000, 1.0) * 0.03
        query_terms = set(re.findall(r"\b\w{4,}\b", query_lower))
        text_terms = set(re.findall(r"\b\w{4,}\b", text_lower))
        term_overlap = len(query_terms & text_terms) / max(len(query_terms), 1) * 0.1

        medical_score = (
            medical_boost
            + section_boost
            + doc_type_boost
            + length_score
            + term_overlap
            - negative_penalty
        )
        return round(min(max(medical_score, 0.0), 0.3), 6)

    def apply(self, query: str, chunks: list[ChunkModel]) -> list[ChunkModel]:
        for chunk in chunks:
            med_score = self.compute_medical_score(chunk, query)
            base = chunk.rerank_score if chunk.rerank_score is not None else (chunk.score / 100.0)
            chunk.rerank_score = round(min(base + med_score, 1.0), 6)
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)


class RerankingService:
    """
    Full reranking pipeline:
      1. BGE Reranker (semantic cross-attention)
      2. Cross-Encoder Reranker (relevance scoring)
      3. Score fusion
      4. Medical domain boost
    """

    def __init__(
        self,
        bge_model: str = "BAAI/bge-reranker-base",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.bge_reranker = BGEReranker(model_name=bge_model)
        self.cross_encoder = CrossEncoderReranker(model_name=cross_encoder_model)
        self.medical_reranker = MedicalReranker()

        available = []
        if self.bge_reranker.is_available:
            available.append("BGE")
        if self.cross_encoder.is_available:
            available.append("CrossEncoder")
        available.append("MedicalDomain")
        logger.info(f"RerankingService initialized | active rerankers: {available}")

    def rerank(
        self,
        query: str,
        chunks: list[ChunkModel],
        top_n: Optional[int] = None,
        use_bge: bool = True,
        use_cross_encoder: bool = True,
        use_medical: bool = True,
    ) -> list[ChunkModel]:
        if not chunks:
            return chunks

        n = top_n or settings.rerank_top_n
        logger.info(f"Reranking {len(chunks)} chunks → top {n}")

        working = list(chunks)

        bge_scores: dict[str, float] = {}
        ce_scores: dict[str, float] = {}

        if use_bge and self.bge_reranker.is_available:
            bge_result = self.bge_reranker.rerank(query, working)
            for c in bge_result:
                bge_scores[c.chunk_id] = c.rerank_score or 0.0

        if use_cross_encoder and self.cross_encoder.is_available:
            ce_result = self.cross_encoder.rerank(query, working)
            for c in ce_result:
                ce_scores[c.chunk_id] = c.rerank_score or 0.0

        has_bge = bool(bge_scores)
        has_ce = bool(ce_scores)

        for chunk in working:
            if has_bge and has_ce:
                bge_s = bge_scores.get(chunk.chunk_id, 0.0)
                ce_s = ce_scores.get(chunk.chunk_id, 0.0)
                chunk.rerank_score = round(0.5 * bge_s + 0.5 * ce_s, 6)
            elif has_bge:
                chunk.rerank_score = bge_scores.get(chunk.chunk_id, 0.0)
            elif has_ce:
                chunk.rerank_score = ce_scores.get(chunk.chunk_id, 0.0)
            else:
                chunk.rerank_score = chunk.score / 100.0

        if use_medical:
            working = self.medical_reranker.apply(query, working)

        working.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        result = working[:n]

        logger.info(
            f"Reranking complete: {len(result)} chunks | "
            f"top_score={result[0].rerank_score:.4f}" if result else "Reranking complete: 0 chunks"
        )
        return result
