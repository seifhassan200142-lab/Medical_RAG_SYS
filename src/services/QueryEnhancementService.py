"""
Query Enhancement: Multi-Query generation and HyDE (Hypothetical Document Embedding).
"""
from groq import Groq
from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.helpers.exceptions import HyDEError

logger = get_logger(__name__)

MULTI_QUERY_SYSTEM = """You are a medical information retrieval expert.
Given a user's medical question, generate {n} semantically diverse alternative phrasings.
These variants should:
- Use different medical terminology and synonyms
- Approach the question from different clinical angles
- Vary between lay terms and technical terminology
- Be suitable for searching a medical knowledge base

Return ONLY the {n} alternative questions, one per line. No numbering, no explanation."""

HYDE_SYSTEM = """You are a medical expert writing a passage that would ideally answer the following question.
Write a concise, factual medical paragraph (3-5 sentences) that a medical textbook or clinical guideline might contain to answer this question.
Use precise medical terminology. Focus on clinical facts, not opinion.
Write ONLY the hypothetical passage, nothing else."""


class QueryEnhancementService:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate_multi_queries(self, question: str, n: int | None = None) -> list[str]:
        count = n or settings.multi_query_count
        logger.info(f"Generating {count} query variants for: '{question[:60]}'")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": MULTI_QUERY_SYSTEM.format(n=count),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            variants = [
                line.strip()
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            variants = variants[:count]
            logger.info(f"Generated {len(variants)} query variants")
            return [question] + variants
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e} — using original query")
            return [question]

    def generate_hyde_document(self, question: str) -> str:
        logger.info(f"Generating HyDE document for: '{question[:60]}'")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=settings.hyde_temperature,
                max_tokens=settings.hyde_max_tokens,
            )
            hyde_text = response.choices[0].message.content or ""
            logger.info(f"HyDE document generated ({len(hyde_text)} chars)")
            return hyde_text.strip()
        except Exception as e:
            raise HyDEError(f"HyDE generation failed: {e}") from e
