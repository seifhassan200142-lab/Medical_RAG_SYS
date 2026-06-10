"""
Context Validation Service
Implements:
  - Context Validation
  - Evidence Verification
  - Unsupported Claim Detection
  - Final Confidence Calibration
"""
import re
from typing import Optional
from dataclasses import dataclass, field

from src.models.ChunkModel import ChunkModel
from src.models.schemas import ConfidenceBreakdown
from src.helpers.config import settings
from src.helpers.logger import get_logger

logger = get_logger(__name__)

MEDICAL_CLAIM_PATTERNS = [
    r"\b\d+(?:\.\d+)?%\s+(?:of\s+patients?|survival|mortality|sensitivity|specificity)\b",
    r"\b(?:dose|dosage)\s+(?:of|is|:)?\s*\d+\s*(?:mg|mcg|ug|g|units?|iu)\b",
    r"\b(?:half.life|t1/2)\s+(?:of|is)?\s*\d+\s*(?:hours?|days?|minutes?|weeks?)\b",
    r"\b\d+\s*(?:mg|mcg|g|ml|units?|iu)\s*(?:per|/)\s*(?:day|kg|dose|hour)\b",
    r"\b(?:sensitivity|specificity|ppv|npv)\s+(?:of|is|:)?\s*\d+(?:\.\d+)?%?\b",
    r"\bgrade\s+[1-4ivIV]+\b",
    r"\bstage\s+[1-4ivIV]+[abcABC]?\b",
    r"\bclass\s+[1-4ivIV]+\b",
    r"\bnormal\s+(?:range|value)\s+(?:is|of|:)?\s*[\d.,\-]+\b",
]

NUMERIC_CLAIM_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*"
    r"(?:%|mg|mcg|g|ml|mmhg|mmol|ng|pg|iu|units?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

UNCERTAINTY_MARKERS = [
    "may", "might", "could", "possibly", "potentially", "appears to",
    "seems to", "likely", "unlikely", "suggests", "estimated",
    "approximately", "roughly", "about", "around", "unclear",
]

DEFINITIVE_MARKERS = [
    "is", "are", "was", "were", "has been", "have been",
    "demonstrates", "shows", "indicates", "confirms", "proves",
    "established", "documented", "proven",
]


@dataclass
class ClaimValidationResult:
    claim_text: str
    is_supported: bool
    support_score: float
    supporting_chunks: list[str] = field(default_factory=list)
    warning: Optional[str] = None


@dataclass
class ContextValidationResult:
    is_valid: bool
    quality_score: float
    coverage_score: float
    consistency_score: float
    warnings: list[str] = field(default_factory=list)
    flagged_gaps: list[str] = field(default_factory=list)


@dataclass
class EvidenceVerificationResult:
    verified_claims: list[ClaimValidationResult] = field(default_factory=list)
    unsupported_claims: list[ClaimValidationResult] = field(default_factory=list)
    support_ratio: float = 0.0
    evidence_strength: str = "weak"


class ContextValidator:
    """Validates the quality and coverage of retrieved context."""

    def validate(self, question: str, chunks: list[ChunkModel]) -> ContextValidationResult:
        if not chunks:
            return ContextValidationResult(
                is_valid=False,
                quality_score=0.0,
                coverage_score=0.0,
                consistency_score=0.0,
                warnings=["No context retrieved — answer will be based on model knowledge only."],
                flagged_gaps=["No relevant documents found in knowledge base."],
            )

        quality_score = self._score_context_quality(chunks)
        coverage_score = self._score_question_coverage(question, chunks)
        consistency_score = self._score_consistency(chunks)
        warnings = []
        gaps = []

        if quality_score < 0.4:
            warnings.append("Retrieved context has low quality scores — results may be unreliable.")
        if coverage_score < 0.3:
            warnings.append("Retrieved context may not fully address the question.")
            gaps.append(f"Question terms poorly covered: '{question[:80]}'")
        if consistency_score < 0.5:
            warnings.append("Conflicting information detected across retrieved sources.")

        doc_ids = {c.document_id for c in chunks}
        if len(doc_ids) == 1:
            warnings.append("All context from a single document — consider broader search.")

        has_recent = any(
            (c.metadata.get("year") or 0) >= 2015 for c in chunks
        )
        if not has_recent:
            warnings.append("Retrieved sources may be outdated (pre-2015).")

        total_text_len = sum(len(c.text) for c in chunks)
        if total_text_len < 200:
            warnings.append("Very short context retrieved — answer depth may be limited.")
            gaps.append("Insufficient context length for comprehensive answer.")

        overall_valid = quality_score >= 0.25 and coverage_score >= 0.15
        return ContextValidationResult(
            is_valid=overall_valid,
            quality_score=round(quality_score, 4),
            coverage_score=round(coverage_score, 4),
            consistency_score=round(consistency_score, 4),
            warnings=warnings,
            flagged_gaps=gaps,
        )

    def _score_context_quality(self, chunks: list[ChunkModel]) -> float:
        scores = [c.score / 100.0 if c.score > 1 else c.score for c in chunks]
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        top = max(scores)
        return round(0.5 * avg + 0.5 * top, 4)

    def _score_question_coverage(self, question: str, chunks: list[ChunkModel]) -> float:
        q_terms = set(re.findall(r"\b\w{4,}\b", question.lower()))
        if not q_terms:
            return 0.5

        all_context = " ".join(c.text.lower() for c in chunks)
        ctx_terms = set(re.findall(r"\b\w{4,}\b", all_context))
        coverage = len(q_terms & ctx_terms) / len(q_terms)
        return round(coverage, 4)

    def _score_consistency(self, chunks: list[ChunkModel]) -> float:
        if len(chunks) <= 1:
            return 1.0

        contradiction_patterns = [
            (r"\bis\b", r"\bis not\b"),
            (r"\beffective\b", r"\bineffective\b"),
            (r"\bincreases?\b", r"\bdecreases?\b"),
            (r"\brecommended\b", r"\bnot recommended\b"),
            (r"\bindicated\b", r"\bcontraindicated\b"),
        ]

        contradiction_count = 0
        texts = [c.text.lower() for c in chunks]
        for pos_pat, neg_pat in contradiction_patterns:
            has_pos = any(re.search(pos_pat, t) for t in texts)
            has_neg = any(re.search(neg_pat, t) for t in texts)
            if has_pos and has_neg:
                contradiction_count += 1

        consistency = 1.0 - min(contradiction_count * 0.1, 0.5)
        return round(consistency, 4)


class EvidenceVerifier:
    """Verifies claims in generated answers against retrieved context."""

    def verify(self, answer: str, chunks: list[ChunkModel]) -> EvidenceVerificationResult:
        claims = self._extract_claims(answer)
        if not claims:
            return EvidenceVerificationResult(
                verified_claims=[],
                unsupported_claims=[],
                support_ratio=1.0,
                evidence_strength="moderate",
            )

        verified = []
        unsupported = []

        for claim in claims:
            result = self._verify_claim(claim, chunks)
            if result.is_supported:
                verified.append(result)
            else:
                unsupported.append(result)

        total = len(claims)
        support_ratio = len(verified) / total if total > 0 else 1.0

        if support_ratio >= 0.85:
            strength = "strong"
        elif support_ratio >= 0.65:
            strength = "moderate"
        elif support_ratio >= 0.40:
            strength = "weak"
        else:
            strength = "insufficient"

        return EvidenceVerificationResult(
            verified_claims=verified,
            unsupported_claims=unsupported,
            support_ratio=round(support_ratio, 4),
            evidence_strength=strength,
        )

    def _extract_claims(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            if NUMERIC_CLAIM_PATTERN.search(sent):
                claims.append(sent)
            elif any(re.search(p, sent, re.IGNORECASE) for p in MEDICAL_CLAIM_PATTERNS):
                claims.append(sent)
            elif any(m in sent.lower() for m in DEFINITIVE_MARKERS[:6]):
                if len(sent.split()) >= 8:
                    claims.append(sent)
        return claims[:20]

    def _verify_claim(self, claim: str, chunks: list[ChunkModel]) -> ClaimValidationResult:
        claim_lower = claim.lower()
        claim_terms = set(re.findall(r"\b\w{4,}\b", claim_lower))
        best_score = 0.0
        supporting = []

        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            chunk_terms = set(re.findall(r"\b\w{4,}\b", chunk_lower))
            if not chunk_terms:
                continue
            overlap = len(claim_terms & chunk_terms) / len(claim_terms) if claim_terms else 0
            if overlap > 0.35:
                supporting.append(chunk.chunk_id)
                best_score = max(best_score, overlap)

        numbers_in_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        if numbers_in_claim:
            numeric_support = any(
                num in chunk.text for chunk in chunks for num in numbers_in_claim
            )
            if not numeric_support:
                best_score *= 0.5

        is_supported = best_score >= 0.35
        warning = None
        if not is_supported and numbers_in_claim:
            warning = f"Numeric claim not found in sources: '{claim[:100]}'"
        elif not is_supported:
            warning = f"Claim may not be supported by retrieved context: '{claim[:100]}'"

        return ClaimValidationResult(
            claim_text=claim[:200],
            is_supported=is_supported,
            support_score=round(best_score, 4),
            supporting_chunks=supporting[:3],
            warning=warning,
        )


class UnsupportedClaimDetector:
    """Detects and flags claims in answers that lack source support."""

    def __init__(self):
        self.verifier = EvidenceVerifier()

    def detect(
        self,
        answer: str,
        chunks: list[ChunkModel],
    ) -> tuple[list[str], list[str]]:
        result = self.verifier.verify(answer, chunks)
        unsupported_warnings = [
            c.warning for c in result.unsupported_claims if c.warning
        ]
        unsupported_texts = [c.claim_text for c in result.unsupported_claims]
        return unsupported_warnings, unsupported_texts


class ConfidenceCalibrator:
    """
    Final confidence calibration integrating:
    - Retrieval confidence
    - Context validation score
    - Evidence verification result
    - Answer grounding
    """

    def calibrate(
        self,
        base_confidence: ConfidenceBreakdown,
        context_validation: ContextValidationResult,
        evidence_verification: EvidenceVerificationResult,
    ) -> ConfidenceBreakdown:
        cv_weight = 0.15
        ev_weight = 0.20

        cv_contribution = context_validation.quality_score * cv_weight
        ev_contribution = evidence_verification.support_ratio * ev_weight

        calibrated_overall = (
            base_confidence.overall * (1.0 - cv_weight - ev_weight)
            + cv_contribution
            + ev_contribution
        )

        if not context_validation.is_valid:
            calibrated_overall *= 0.7

        evidence_penalties = {
            "insufficient": 0.25,
            "weak": 0.10,
            "moderate": 0.0,
            "strong": 0.0,
        }
        penalty = evidence_penalties.get(evidence_verification.evidence_strength, 0.0)
        calibrated_overall = max(0.0, calibrated_overall - penalty)
        calibrated_overall = round(min(max(calibrated_overall, 0.0), 1.0), 4)

        calibrated_hallucination = base_confidence.hallucination_risk
        if evidence_verification.unsupported_claims:
            extra_risk = min(len(evidence_verification.unsupported_claims) * 0.08, 0.3)
            calibrated_hallucination = round(min(base_confidence.hallucination_risk + extra_risk, 1.0), 4)

        all_warnings = list(base_confidence.warnings)
        all_warnings.extend(context_validation.warnings)
        if evidence_verification.unsupported_claims:
            all_warnings.append(
                f"{len(evidence_verification.unsupported_claims)} claim(s) could not be verified in retrieved sources."
            )
        if evidence_verification.evidence_strength in ("weak", "insufficient"):
            all_warnings.append(
                f"Evidence strength: {evidence_verification.evidence_strength}. "
                "Cross-check with primary medical sources recommended."
            )

        return ConfidenceBreakdown(
            overall=calibrated_overall,
            retrieval_confidence=base_confidence.retrieval_confidence,
            answer_grounding=base_confidence.answer_grounding,
            source_coverage=base_confidence.source_coverage,
            hallucination_risk=calibrated_hallucination,
            is_reliable=calibrated_overall >= settings.confidence_threshold,
            warnings=list(dict.fromkeys(all_warnings)),
        )


class ContextValidationService:
    """
    Unified service combining all validation and calibration steps.
    """

    def __init__(self):
        self.context_validator = ContextValidator()
        self.evidence_verifier = EvidenceVerifier()
        self.unsupported_detector = UnsupportedClaimDetector()
        self.confidence_calibrator = ConfidenceCalibrator()

    def validate_and_calibrate(
        self,
        question: str,
        answer: str,
        chunks: list[ChunkModel],
        base_confidence: ConfidenceBreakdown,
    ) -> tuple[ConfidenceBreakdown, ContextValidationResult, EvidenceVerificationResult, list[str]]:
        context_validation = self.context_validator.validate(question, chunks)
        evidence_verification = self.evidence_verifier.verify(answer, chunks)
        unsupported_warnings, _ = self.unsupported_detector.detect(answer, chunks)
        calibrated_confidence = self.confidence_calibrator.calibrate(
            base_confidence=base_confidence,
            context_validation=context_validation,
            evidence_verification=evidence_verification,
        )

        logger.info(
            f"Validation complete | context_valid={context_validation.is_valid} | "
            f"evidence_strength={evidence_verification.evidence_strength} | "
            f"calibrated_confidence={calibrated_confidence.overall:.3f} | "
            f"unsupported_claims={len(evidence_verification.unsupported_claims)}"
        )

        return calibrated_confidence, context_validation, evidence_verification, unsupported_warnings
