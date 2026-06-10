"""
Confidence scoring and hallucination risk estimation for medical RAG.
"""
import re
from src.models.ChunkModel import ChunkModel
from src.models.schemas import ConfidenceBreakdown
from src.helpers.config import settings
from src.helpers.logger import get_logger

logger = get_logger(__name__)

UNCERTAINTY_PHRASES = [
    "i don't know", "i am not sure", "i cannot", "no information",
    "not found", "unable to", "insufficient", "context does not",
    "cannot determine", "no relevant", "based on my training",
    "as an ai", "i'm not able",
]

HIGH_CONFIDENCE_ANCHORS = [
    "according to", "the study shows", "evidence indicates",
    "guidelines recommend", "research demonstrates", "clinical trials",
    "the findings", "data suggests", "results show",
]

MEDICAL_HEDGE_PHRASES = [
    "may", "might", "could", "possibly", "potentially", "appears",
    "seems", "likely", "unlikely", "suggests", "estimated",
]


def score_retrieval_confidence(chunks: list[ChunkModel]) -> float:
    if not chunks:
        return 0.0
    scores = [c.score for c in chunks]
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)
    coverage = min(len(chunks) / settings.top_k, 1.0)
    return round(0.4 * top_score + 0.4 * avg_score + 0.2 * coverage, 4)


def score_answer_grounding(answer: str, chunks: list[ChunkModel]) -> float:
    if not chunks or not answer:
        return 0.0

    answer_lower = answer.lower()
    total_overlap = 0.0

    for chunk in chunks:
        chunk_words = set(re.findall(r"\b\w{4,}\b", chunk.text.lower()))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer_lower))
        if not chunk_words:
            continue
        overlap = len(chunk_words & answer_words) / len(chunk_words)
        total_overlap += overlap

    grounding = total_overlap / len(chunks)
    return round(min(grounding * 2.0, 1.0), 4)


def estimate_hallucination_risk(answer: str, chunks: list[ChunkModel]) -> float:
    answer_lower = answer.lower()

    uncertainty_hits = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase in answer_lower)
    hedge_hits = sum(1 for phrase in MEDICAL_HEDGE_PHRASES if phrase in answer_lower)
    anchor_hits = sum(1 for phrase in HIGH_CONFIDENCE_ANCHORS if phrase in answer_lower)

    if not chunks:
        return 0.9

    avg_score = sum(c.score for c in chunks) / len(chunks)
    base_risk = max(0.0, 0.8 - avg_score)

    risk = base_risk
    risk += uncertainty_hits * 0.1
    risk += hedge_hits * 0.02
    risk -= anchor_hits * 0.05

    return round(min(max(risk, 0.0), 1.0), 4)


def score_source_coverage(chunks: list[ChunkModel]) -> float:
    if not chunks:
        return 0.0
    doc_ids = {c.document_id for c in chunks}
    sections = {c.section for c in chunks if c.section and c.section != "general"}
    doc_diversity = min(len(doc_ids) / 3.0, 1.0)
    section_diversity = min(len(sections) / 4.0, 1.0)
    return round(0.6 * doc_diversity + 0.4 * section_diversity, 4)


def compute_confidence(
    answer: str,
    chunks: list[ChunkModel],
) -> ConfidenceBreakdown:
    retrieval_conf = score_retrieval_confidence(chunks)
    grounding = score_answer_grounding(answer, chunks)
    coverage = score_source_coverage(chunks)
    hallucination_risk = estimate_hallucination_risk(answer, chunks)

    overall = (
        0.35 * retrieval_conf
        + 0.35 * grounding
        + 0.15 * coverage
        + 0.15 * (1.0 - hallucination_risk)
    )
    overall = round(min(max(overall, 0.0), 1.0), 4)

    warnings = []
    if retrieval_conf < 0.4:
        warnings.append("Low retrieval confidence — answer may not be well-supported.")
    if grounding < 0.3:
        warnings.append("Answer may contain information not found in retrieved sources.")
    if hallucination_risk > settings.hallucination_threshold:
        warnings.append("Elevated hallucination risk detected — verify with primary sources.")
    if not chunks:
        warnings.append("No context retrieved — answer based on model knowledge only.")

    return ConfidenceBreakdown(
        overall=overall,
        retrieval_confidence=retrieval_conf,
        answer_grounding=grounding,
        source_coverage=coverage,
        hallucination_risk=hallucination_risk,
        is_reliable=overall >= settings.confidence_threshold,
        warnings=warnings,
    )
