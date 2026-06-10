"""
Medical Hybrid Retrieval Service.
Pipeline: Query Enhancement → HyDE → Hybrid Search (Dense + BM25/Sparse)
         → Reciprocal Rank Fusion → Parent Retrieval → Context Compression
"""
from qdrant_client.models import ScoredPoint

from src.models.ChunkModel import ChunkModel
from src.services.EmbeddingService import EmbeddingService
from src.services.QueryEnhancementService import QueryEnhancementService
from src.stores.QdrantStore import QdrantStore
from src.medical.bm25_encoder import BM25Encoder
from src.helpers.config import settings
from src.helpers.logger import get_logger
from src.helpers.exceptions import RetrievalError

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    result_lists: list[list[ScoredPoint]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranked_list in result_lists:
        for rank, point in enumerate(ranked_list, start=1):
            pid = str(point.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in payloads and point.payload:
                payloads[pid] = point.payload

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(pid, scores[pid], payloads.get(pid, {})) for pid in sorted_ids]


def _compress_context(chunks: list[ChunkModel], ratio: float = 0.6) -> list[ChunkModel]:
    """
    Context compression: keep top chunks by score until we hit ratio of top-k.
    Simple score-based selection that removes low-value chunks.
    """
    if not chunks:
        return chunks
    target = max(1, int(len(chunks) * ratio))
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    return sorted_chunks[:target]


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_store: QdrantStore,
        query_enhancement: QueryEnhancementService,
        bm25_encoder: BM25Encoder,
    ):
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.query_enhancement = query_enhancement
        self.bm25_encoder = bm25_encoder

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
        specialty: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        use_parent_retrieval: bool = True,
        use_compression: bool = True,
    ) -> tuple[list[ChunkModel], list[str], str | None]:
        """
        Full retrieval pipeline.
        Returns: (chunks, query_variants, hyde_hypothesis)
        """
        k = top_k or settings.top_k

        query_filter = self.qdrant_store.build_filter(
            document_id=document_id,
            document_type=document_type,
            specialty=specialty,
            year_from=year_from,
            year_to=year_to,
        )

        # 1. Multi-Query generation
        if use_multi_query:
            query_variants = self.query_enhancement.generate_multi_queries(question)
        else:
            query_variants = [question]

        # 2. HyDE
        hyde_hypothesis = None
        if use_hyde:
            try:
                hyde_hypothesis = self.query_enhancement.generate_hyde_document(question)
            except Exception as e:
                logger.warning(f"HyDE skipped: {e}")

        # 3. Collect all result lists for RRF
        all_dense_results: list[list[ScoredPoint]] = []
        all_sparse_results: list[list[ScoredPoint]] = []

        for query in query_variants:
            dense_vec = self.embedding_service.embed_query(query)
            dense_results = self.qdrant_store.dense_search(
                query_vector=dense_vec,
                top_k=k * 2,
                score_threshold=settings.score_threshold,
                query_filter=query_filter,
            )
            all_dense_results.append(dense_results)

            sparse_indices, sparse_values = self.bm25_encoder.encode_query(query)
            if sparse_indices:
                sparse_results = self.qdrant_store.sparse_search(
                    indices=sparse_indices,
                    values=sparse_values,
                    top_k=k * 2,
                    query_filter=query_filter,
                )
                all_sparse_results.append(sparse_results)

        # 4. HyDE dense search
        if hyde_hypothesis:
            hyde_vec = self.embedding_service.embed_query(hyde_hypothesis)
            hyde_results = self.qdrant_store.dense_search(
                query_vector=hyde_vec,
                top_k=k,
                score_threshold=settings.score_threshold,
                query_filter=query_filter,
            )
            all_dense_results.append(hyde_results)

        # 5. Reciprocal Rank Fusion across all result lists
        merged = _reciprocal_rank_fusion(all_dense_results + all_sparse_results)

        if not merged:
            logger.warning("No results after hybrid retrieval")
            return [], query_variants, hyde_hypothesis

        # 6. Build ChunkModel list from fused results
        top_merged = merged[:k * 2]
        chunks: list[ChunkModel] = []

        for pid, rrf_score, payload in top_merged:
            if not payload:
                continue
            chunk = ChunkModel(
                chunk_id=payload.get("chunk_id", pid),
                document_id=payload.get("document_id", ""),
                parent_chunk_id=payload.get("parent_chunk_id"),
                text=payload.get("text", ""),
                metadata={
                    "filename": payload.get("filename", ""),
                    "page": payload.get("page", 0),
                    "section": payload.get("section", ""),
                    "document_type": payload.get("document_type", ""),
                    "specialty": payload.get("specialty"),
                    "year": payload.get("year"),
                    "authors": payload.get("authors"),
                    "journal": payload.get("journal"),
                    "source": payload.get("source", ""),
                },
                score=round(rrf_score * 100, 4),
                section=payload.get("section"),
                document_type=payload.get("document_type"),
            )
            chunks.append(chunk)

        # 7. Parent Document Retrieval
        if use_parent_retrieval and chunks:
            chunks = self._expand_to_parents(chunks, k)

        # 8. Context Compression
        if use_compression and chunks:
            chunks = _compress_context(chunks, settings.context_compression_ratio)

        final_chunks = chunks[:k]
        logger.info(
            f"Retrieval complete: {len(final_chunks)} chunks | "
            f"variants={len(query_variants)} | hyde={'yes' if hyde_hypothesis else 'no'}"
        )
        return final_chunks, query_variants, hyde_hypothesis

    def _expand_to_parents(
        self,
        child_chunks: list[ChunkModel],
        top_k: int,
    ) -> list[ChunkModel]:
        """
        Replace high-scoring child chunks with their parent chunks for richer context.
        Keeps children for lower-scoring results.
        """
        parent_candidate_ids = []
        parent_score_map: dict[str, float] = {}

        top_children = sorted(child_chunks, key=lambda c: c.score, reverse=True)
        top_n = max(1, settings.parent_top_k)

        for chunk in top_children[:top_n]:
            if chunk.parent_chunk_id:
                parent_candidate_ids.append(chunk.parent_chunk_id)
                parent_score_map[chunk.parent_chunk_id] = chunk.score

        if not parent_candidate_ids:
            return child_chunks

        try:
            parent_records = self.qdrant_store.get_parents_by_ids(parent_candidate_ids)
        except Exception as e:
            logger.warning(f"Parent retrieval failed: {e} — using child chunks")
            return child_chunks

        parent_chunks = []
        for record in parent_records:
            payload = record.payload or {}
            parent_id = str(record.id)
            parent_chunk = ChunkModel(
                chunk_id=parent_id,
                document_id=payload.get("document_id", ""),
                parent_chunk_id=parent_id,
                text=payload.get("text", ""),
                metadata={
                    "filename": payload.get("filename", ""),
                    "page": payload.get("page", 0),
                    "section": payload.get("section", ""),
                    "document_type": payload.get("document_type", ""),
                    "specialty": payload.get("specialty"),
                    "year": payload.get("year"),
                    "authors": payload.get("authors"),
                    "journal": payload.get("journal"),
                },
                score=parent_score_map.get(parent_id, 0.0),
                section=payload.get("section"),
                document_type=payload.get("document_type"),
            )
            parent_chunks.append(parent_chunk)

        replaced_parent_ids = {str(r.id) for r in parent_records}
        remaining_children = [
            c for c in child_chunks
            if c.parent_chunk_id not in replaced_parent_ids
        ]

        combined = parent_chunks + remaining_children
        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[:top_k * 2]
