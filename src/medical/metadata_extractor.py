import re
from typing import Any
from src.helpers.config import settings
from src.helpers.logger import get_logger

logger = get_logger(__name__)

DOCUMENT_TYPE_PATTERNS = {
    "research_paper": [
        r"\babstract\b", r"\bintroduction\b", r"\bmethods?\b", r"\bresults?\b",
        r"\bdiscussion\b", r"\bconclusion\b", r"\breferences\b", r"\bdoi\b",
        r"\bjournal\b", r"\bpublished\b", r"\bcited by\b",
    ],
    "clinical_note": [
        r"\bpatient\b", r"\bchief complaint\b", r"\bhistory of present illness\b",
        r"\bphysical examination\b", r"\bassessment\b", r"\bplan\b",
        r"\bmedications?\b", r"\ballergies\b", r"\bvital signs\b",
        r"\bdiagnosis\b", r"\bfollow.?up\b",
    ],
    "lab_report": [
        r"\blab(?:oratory)? results?\b", r"\btest results?\b", r"\bnormal range\b",
        r"\breference range\b", r"\bspecimen\b", r"\bcollected\b",
        r"\bcbc\b", r"\bwbc\b", r"\brbc\b", r"\bglucose\b", r"\bcreatinine\b",
        r"\bhmg\b", r"\bplt\b",
    ],
    "medical_guideline": [
        r"\bguideline\b", r"\brecommendation\b", r"\bevidence.?based\b",
        r"\bgrade [abcd]\b", r"\bstrong recommendation\b", r"\bconditional\b",
        r"\bclinical practice\b", r"\bstandard of care\b",
    ],
    "radiology_report": [
        r"\bfinding\b", r"\bimpression\b", r"\bradiology\b", r"\bimaging\b",
        r"\bmri\b", r"\bct scan\b", r"\bx.?ray\b", r"\bultrasound\b",
        r"\bcontrast\b", r"\bnodule\b", r"\blesion\b",
    ],
    "discharge_summary": [
        r"\bdischarge\b", r"\badmission\b", r"\bhospital\b",
        r"\bdischarge diagnosis\b", r"\bdischarge medications?\b",
        r"\bfollow.?up instructions?\b",
    ],
}

MEDICAL_SPECIALTIES = [
    "cardiology", "oncology", "neurology", "orthopedics", "radiology",
    "pathology", "psychiatry", "endocrinology", "gastroenterology",
    "pulmonology", "nephrology", "rheumatology", "hematology", "infectious",
    "pediatrics", "geriatrics", "obstetrics", "gynecology", "urology",
    "dermatology", "ophthalmology", "ent", "emergency", "surgery",
]

SECTION_PATTERNS = {
    "abstract": [r"^abstract$", r"^summary$"],
    "introduction": [r"^introduction$", r"^background$", r"^overview$"],
    "methods": [r"^methods?$", r"^methodology$", r"^materials? and methods?$"],
    "results": [r"^results?$", r"^findings?$", r"^outcomes?$"],
    "discussion": [r"^discussion$", r"^analysis$"],
    "conclusion": [r"^conclusions?$", r"^concluding remarks$"],
    "references": [r"^references?$", r"^bibliography$", r"^citations?$"],
    "history": [r"^(history|hpi|history of present illness)$"],
    "examination": [r"^(physical exam|examination|pe)$"],
    "assessment": [r"^(assessment|impression|diagnosis)$"],
    "plan": [r"^(plan|treatment|management|recommendations?)$"],
    "medications": [r"^(medications?|current medications?|drug list)$"],
    "labs": [r"^(laboratory|lab results?|labs?)$"],
    "imaging": [r"^(imaging|radiology|radiologic findings?)$"],
}


def detect_document_type(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        scores[doc_type] = score

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def detect_specialty(text: str) -> str | None:
    text_lower = text.lower()
    for spec in MEDICAL_SPECIALTIES:
        if spec in text_lower:
            return spec
    return None


def extract_authors(text: str) -> str | None:
    patterns = [
        r"(?:authors?|by)\s*:?\s*([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)",
        r"^([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+){1,5})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def extract_journal(text: str) -> str | None:
    patterns = [
        r"(?:journal|published in|in)\s*:?\s*([A-Z][A-Za-z\s&]+(?:Journal|Medicine|Review|Research|Science|Health))",
        r"([A-Z][A-Za-z\s]+(?:Journal|Medicine|Review|Research))\s*\d{4}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:3000])
        if match:
            return match.group(1).strip()
    return None


def extract_year(text: str) -> int | None:
    match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text[:3000])
    return int(match.group()) if match else None


def detect_section(line: str) -> str | None:
    stripped = line.strip().lower()
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return section
    return None


def extract_document_metadata(text: str, filename: str) -> dict[str, Any]:
    doc_type = detect_document_type(text)
    specialty = detect_specialty(text)
    year = extract_year(text)
    authors = extract_authors(text)
    journal = extract_journal(text)

    return {
        "document_type": doc_type,
        "specialty": specialty,
        "year": year,
        "authors": authors,
        "journal": journal,
        "filename": filename,
        "source": filename,
    }


def assign_section_to_chunks(chunks: list[dict]) -> list[dict]:
    current_section = "general"
    for chunk in chunks:
        text = chunk.get("text", "")
        first_lines = text.split("\n")[:3]
        for line in first_lines:
            detected = detect_section(line)
            if detected:
                current_section = detected
                break
        chunk["section"] = current_section
    return chunks
