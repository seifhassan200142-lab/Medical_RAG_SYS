from langchain.prompts import ChatPromptTemplate

MEDICAL_RAG_SYSTEM = """You are a specialized medical AI assistant with expertise in clinical medicine, pharmacology, and medical research.

Your task is to answer the medical question using ONLY the provided context from medical documents.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided context. Do not use external knowledge.
2. If the context is insufficient, clearly state: "The provided documents do not contain sufficient information to answer this question."
3. Cite specific sources using [N] notation where N corresponds to the source number in context.
4. Use precise medical terminology. Spell out abbreviations on first use.
5. If information conflicts between sources, note the discrepancy.
6. Never fabricate clinical values, drug doses, or diagnostic criteria.
7. Structure complex answers with clear sections when appropriate.
8. Always answer in the exact same language as the user's question. You MUST translate the relevant information from the retrieved context accurately to the user's language, without adding any external clinical knowledge.
Context:
{context}"""

MEDICAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MEDICAL_RAG_SYSTEM),
    ("human", "{question}"),
])


def build_medical_prompt(context: str, question: str) -> list[dict]:
    messages = MEDICAL_RAG_PROMPT.format_messages(context=context, question=question)
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


def build_context_block(chunks) -> str:
    """Build numbered context block with source metadata."""
    if not chunks:
        return "No relevant context retrieved."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        filename = meta.get("filename", "unknown")
        page = meta.get("page", "")
        section = meta.get("section", "")
        doc_type = meta.get("document_type", "")
        year = meta.get("year", "")
        authors = meta.get("authors", "")

        header_parts = [f"[{i}] {filename}"]
        if doc_type:
            header_parts.append(f"({doc_type})")
        if authors:
            header_parts.append(f"| {authors}")
        if year:
            header_parts.append(f"| {year}")
        if page:
            header_parts.append(f"| p.{page}")
        if section:
            header_parts.append(f"| §{section}")

        header = " ".join(header_parts)
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)
