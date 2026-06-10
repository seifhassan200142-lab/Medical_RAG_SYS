"""
Medical section-aware chunking.
Splits documents while preserving section boundaries and medical context.
"""
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.helpers.config import settings
from src.medical.metadata_extractor import detect_section
from src.helpers.logger import get_logger

logger = get_logger(__name__)

SECTION_BREAK_PATTERNS = [
    r"^\s{0,4}(ABSTRACT|INTRODUCTION|BACKGROUND|METHODS?|RESULTS?|DISCUSSION|CONCLUSION|REFERENCES?)\s*$",
    r"^\s{0,4}(Chief Complaint|History of Present Illness|Physical Examination|Assessment|Plan|Medications?|Allergies|Vital Signs)\s*:?\s*$",
    r"^\s{0,4}(Findings?|Impression|Clinical History|Laboratory Results?)\s*:?\s*$",
    r"^#{1,3}\s+",
    r"^\d+\.\s+[A-Z]",
]

SECTION_BREAK_RE = re.compile(
    "|".join(f"(?:{p})" for p in SECTION_BREAK_PATTERNS),
    re.MULTILINE | re.IGNORECASE,
)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, section_text) pairs."""
    positions = [(m.start(), m.group().strip()) for m in SECTION_BREAK_RE.finditer(text)]

    if not positions:
        return [("general", text)]

    sections = []
    prev_pos = 0
    prev_name = "preamble"

    for pos, header in positions:
        chunk_text = text[prev_pos:pos].strip()
        if chunk_text:
            sections.append((prev_name, chunk_text))
        prev_pos = pos
        prev_name = detect_section(header) or header.strip().lower()

    remaining = text[prev_pos:].strip()
    if remaining:
        sections.append((prev_name, remaining))

    return sections


def medical_chunk_documents(
    documents: list[Document],
    child_size: int | None = None,
    child_overlap: int | None = None,
    parent_size: int | None = None,
    parent_overlap: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (child_chunks, parent_chunks).
    Each child dict: {text, section, metadata, parent_chunk_id}
    Each parent dict: {text, section, metadata, parent_chunk_id}
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size or settings.chunk_size,
        chunk_overlap=child_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size or settings.parent_chunk_size,
        chunk_overlap=parent_overlap or settings.parent_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_chunks = []
    parent_chunks = []

    for doc in documents:
        base_meta = doc.metadata.copy()
        page = base_meta.get("page", 0)
        text = doc.page_content

        sections = _split_into_sections(text)

        for section_name, section_text in sections:
            if not section_text.strip():
                continue

            parent_splits = parent_splitter.split_text(section_text)
            for parent_text in parent_splits:
                if not parent_text.strip():
                    continue

                import uuid
                parent_id = str(uuid.uuid4())
                parent_chunks.append({
                    "parent_chunk_id": parent_id,
                    "text": parent_text,
                    "section": section_name,
                    "metadata": {**base_meta, "page": page, "section": section_name},
                })

                child_splits = child_splitter.split_text(parent_text)
                for child_text in child_splits:
                    if not child_text.strip():
                        continue
                    child_chunks.append({
                        "parent_chunk_id": parent_id,
                        "text": child_text,
                        "section": section_name,
                        "metadata": {**base_meta, "page": page, "section": section_name},
                    })

    logger.info(f"Section-aware chunking: {len(child_chunks)} child, {len(parent_chunks)} parent chunks")
    return child_chunks, parent_chunks
