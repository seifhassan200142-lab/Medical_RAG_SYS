"""
Generation Service with integrated context validation, evidence verification,
unsupported claim detection, and final confidence calibration.
"""
from groq import Groq

from src.models.ChunkModel import ChunkModel
from src.models.schemas import Citation, ConfidenceBreakdown
from src.langchain_components.prompts import build_medical_prompt, build_context_block
from src.medical.citation_generator import generate_citations, inject_citation_markers, format_citation_block
from src.medical.confidence_scorer import compute_confidence
from src.services.ContextValidationService import ContextValidationService
from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.helpers.exceptions import GenerationError

logger = get_logger(__name__)


class GenerationService:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        self.validation_service = ContextValidationService()

    def generate(
        self,
        question: str,
        chunks: list[ChunkModel],
    ) -> tuple[str, list[Citation], ConfidenceBreakdown]:
        context = build_context_block(chunks)
        messages = build_medical_prompt(context=context, question=question)

        logger.info(f"Calling {self.model} | context_chunks={len(chunks)}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            raise GenerationError(f"Groq API call failed: {e}") from e

        raw_answer = response.choices[0].message.content or ""
        logger.info(f"Generated answer: {len(raw_answer)} chars")

        citations = generate_citations(chunks)
        answer_with_markers = inject_citation_markers(raw_answer, chunks)
        reference_block = format_citation_block(citations)
        final_answer = answer_with_markers + reference_block

        base_confidence = compute_confidence(raw_answer, chunks)

        calibrated_confidence, context_validation, evidence_verification, unsupported_warnings = (
            self.validation_service.validate_and_calibrate(
                question=question,
                answer=raw_answer,
                chunks=chunks,
                base_confidence=base_confidence,
            )
        )

        if unsupported_warnings:
            unsupported_block = "\n\n⚠️ **Unverified Claims Detected:**\n" + "\n".join(
                f"- {w}" for w in unsupported_warnings[:5]
            )
            final_answer += unsupported_block

        if not calibrated_confidence.is_reliable:
            disclaimer = (
                "\n\n⚠️ **Reliability Warning**: "
                f"Confidence score {calibrated_confidence.overall:.2f} is below threshold. "
                "Please verify this information with primary medical sources."
            )
            final_answer += disclaimer

        logger.info(
            f"Confidence (calibrated): {calibrated_confidence.overall:.2f} | "
            f"Hallucination risk: {calibrated_confidence.hallucination_risk:.2f} | "
            f"Evidence strength: {evidence_verification.evidence_strength} | "
            f"Citations: {len(citations)}"
        )

        return final_answer, citations, calibrated_confidence, context_validation, evidence_verification
