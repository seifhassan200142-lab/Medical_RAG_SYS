"""
Citation generation for medical RAG responses.
Produces structured, numbered citations from retrieved chunks.
"""
import re
from src.models.ChunkModel import ChunkModel
from src.models.schemas import Citation
from src.helpers.logger import get_logger

logger = get_logger(__name__)


def generate_citations(chunks: list[ChunkModel]) -> list[Citation]:
    """Convert ranked chunks to numbered Citation objects."""
    citations = []
    seen_ids = set()

    for i, chunk in enumerate(chunks, start=1):
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)

        meta = chunk.metadata or {}
        citation = Citation(
            citation_number=i,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=meta.get("filename", "unknown"),
            page=meta.get("page"),
            section=meta.get("section") or chunk.section,
            document_type=meta.get("document_type") or chunk.document_type,
            authors=meta.get("authors"),
            year=meta.get("year"),
            journal=meta.get("journal"),
            score=round(chunk.score, 4),
            text_preview=chunk.text[:250].strip(),
        )
        citations.append(citation)

    return citations


def inject_citation_markers(answer: str, chunks: list[ChunkModel]) -> str:
    """
    Post-process an LLM answer to add [N] citation markers.
    Matches key phrases from chunks to the answer text.
    """
    if not chunks:
        return answer

    marked_answer = answer
    for i, chunk in enumerate(chunks, start=1):
        key_phrases = _extract_key_phrases(chunk.text)
        for phrase in key_phrases:
            if phrase.lower() in marked_answer.lower() and f"[{i}]" not in marked_answer:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                marked_answer = pattern.sub(f"{phrase} [{i}]", marked_answer, count=1)
                break

    return marked_answer


def _extract_key_phrases(text: str, max_phrases: int = 3) -> list[str]:
    """Extract short key phrases likely to appear verbatim in LLM output."""
    sentences = re.split(r"[.!?]", text)
    phrases = []
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if 4 <= len(words) <= 15:
            phrases.append(sent)
        if len(phrases) >= max_phrases:
            break
    return phrases


def format_citation_block(citations: list[Citation]) -> str:
    """Format a references block for appending to LLM answers."""
    if not citations:
        return ""

    lines = ["\n\n**References:**"]
    for c in citations:
        parts = [f"[{c.citation_number}]"]
        if c.authors:
            parts.append(c.authors)
        parts.append(c.filename)
        if c.year:
            parts.append(f"({c.year})")
        if c.journal:
            parts.append(f"*{c.journal}*")
        if c.page:
            parts.append(f"p. {c.page}")
        if c.section:
            parts.append(f"§ {c.section}")
        lines.append(" — ".join(parts))

    return "\n".join(lines)
