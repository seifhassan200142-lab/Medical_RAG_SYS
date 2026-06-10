import pytest
from unittest.mock import MagicMock, patch
from src.medical.confidence_scorer import (
    score_retrieval_confidence,
    score_answer_grounding,
    estimate_hallucination_risk,
    compute_confidence,
)
from src.models.ChunkModel import ChunkModel


def make_chunk(score=0.8, text="Aspirin reduces platelet aggregation via COX inhibition."):
    return ChunkModel(
        chunk_id="test-chunk-id",
        document_id="test-doc-id",
        text=text,
        score=score,
        section="methods",
        document_type="research_paper",
        metadata={"filename": "test.pdf", "page": 1},
    )


def test_retrieval_confidence_empty():
    assert score_retrieval_confidence([]) == 0.0


def test_retrieval_confidence_with_chunks():
    chunks = [make_chunk(0.9), make_chunk(0.7), make_chunk(0.8)]
    score = score_retrieval_confidence(chunks)
    assert 0.0 < score <= 1.0


def test_answer_grounding():
    chunks = [make_chunk(text="Aspirin reduces platelet aggregation via COX inhibition.")]
    grounding = score_answer_grounding(
        "Aspirin works by reducing platelet aggregation.", chunks
    )
    assert grounding > 0.0


def test_hallucination_risk_no_chunks():
    risk = estimate_hallucination_risk("Some answer", [])
    assert risk > 0.5


def test_full_confidence():
    chunks = [make_chunk(0.85), make_chunk(0.75)]
    answer = "Aspirin reduces platelet aggregation which is used in treatment."
    result = compute_confidence(answer, chunks)
    assert 0.0 <= result.overall <= 1.0
    assert isinstance(result.is_reliable, bool)
    assert isinstance(result.warnings, list)
