"""
BM25 sparse encoder using rank-bm25 for sparse vector generation.
Converts text to sparse TF-IDF-like vectors compatible with Qdrant sparse search.
"""
import math
import re
from collections import Counter
from typing import Optional
from src.helpers.logger import get_logger

logger = get_logger(__name__)

MEDICAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "it", "its", "as", "not",
}

MEDICAL_TERM_BOOST = {
    "diagnosis", "treatment", "symptoms", "medication", "dosage", "therapy",
    "patient", "clinical", "disease", "syndrome", "disorder", "condition",
    "laboratory", "imaging", "biopsy", "pathology", "prognosis",
    "contraindication", "adverse", "efficacy", "protocol", "guideline",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z][a-z0-9\-]{1,}\b", text)
    return [t for t in tokens if t not in MEDICAL_STOPWORDS and len(t) > 1]


def build_vocab(corpus: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in corpus:
        for token in set(tokenize(text)):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


class BM25Encoder:
    """
    BM25 encoder that produces sparse vectors (indices + values)
    compatible with Qdrant's sparse vector format.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        logger.info(f"Fitting BM25 encoder on {len(corpus)} documents")
        self.vocab = build_vocab(corpus)
        self.doc_count = len(corpus)

        tokenized = [tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(t) for t in tokenized) / max(self.doc_count, 1)

        df: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            self.idf[token] = math.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )

        self._fitted = True
        logger.info(f"BM25 vocab size: {len(self.vocab)}")

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        if not self._fitted:
            return self._fallback_encode(text)

        tokens = tokenize(text)
        token_freq = Counter(tokens)
        dl = len(tokens)

        indices = []
        values = []

        for token, tf in token_freq.items():
            if token not in self.vocab:
                continue
            idf = self.idf.get(token, 0.5)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score = idf * numerator / denominator

            boost = 1.5 if token in MEDICAL_TERM_BOOST else 1.0
            indices.append(self.vocab[token])
            values.append(float(score * boost))

        return indices, values

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        return self.encode_query(text)

    def _fallback_encode(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        token_freq = Counter(tokens)
        total = sum(token_freq.values()) or 1

        indices = []
        values = []
        for i, (token, freq) in enumerate(token_freq.most_common(64)):
            indices.append(hash(token) % 100000)
            values.append(float(freq) / total)

        return indices, values

    def update(self, new_docs: list[str]) -> None:
        if not self._fitted:
            self.fit(new_docs)
            return

        for text in new_docs:
            for token in set(tokenize(text)):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
