"""
Ingestion Service — Unified PDF Multimodal Pipeline

For PDFs:
  PDF → Extract Text + Extract Tables + Extract Images (PyMuPDF)
  Text  → Medical Chunking → Embeddings
  Tables → Table Summary → Embeddings
  Images → MedicalImageService → OCR → VLM Analysis → Findings → Caption → Embeddings
  All chunks → Qdrant (same collection, chunk_type field differentiates)

For TXT/MD: text-only pipeline (unchanged).
"""
import os
from qdrant_client.models import PointStruct, SparseVector

from src.services.EmbeddingService import EmbeddingService
from src.services.MedicalImageService import MedicalImageService
from src.services.PDFImageExtractor import PDFImageExtractor
from src.stores.QdrantStore import QdrantStore
from src.stores.DocumentStore import DocumentStore
from src.models.DocumentModel import DocumentModel
from src.medical.metadata_extractor import extract_document_metadata
from src.medical.section_splitter import medical_chunk_documents
from src.medical.bm25_encoder import BM25Encoder
from src.langchain_components.loaders import load_pdf, load_text, load_markdown
from src.helpers.utils import generate_id, file_checksum, now_utc, get_file_extension, clean_text
from src.helpers.exceptions import IngestionError, UnsupportedFileTypeError
from src.helpers.logger import get_logger
from src.helpers.config import settings

logger = get_logger(__name__)

SUPPORTED_TYPES = {"pdf", "txt", "md"}
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# chunk_type field values stored in Qdrant payloads
CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_TABLE = "table"
CHUNK_TYPE_IMAGE = "image"


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_store: QdrantStore,
        document_store: DocumentStore,
        bm25_encoder: BM25Encoder,
        image_service: MedicalImageService,
    ):
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.document_store = document_store
        self.bm25_encoder = bm25_encoder
        self.image_service = image_service
        self.pdf_extractor = PDFImageExtractor()

    def process_document(self, file_path: str, original_filename: str) -> DocumentModel:
        ext = get_file_extension(original_filename)
        if ext not in SUPPORTED_TYPES:
            raise UnsupportedFileTypeError(ext)

        checksum = file_checksum(file_path)
        existing = self.document_store.exists_by_checksum(checksum)
        if existing:
            logger.info(f"Duplicate detected (checksum match): {existing.id}")
            return existing

        if ext == "pdf":
            return self._process_pdf(file_path, original_filename, checksum)
        else:
            raw_docs = load_text(file_path) if ext == "txt" else load_markdown(file_path)
            return self._process_text_docs(raw_docs, file_path, original_filename, ext, checksum)

    # ── PDF Pipeline ──────────────────────────────────────────────────────────

    def _process_pdf(self, file_path: str, filename: str, checksum: str) -> DocumentModel:
        document_id = generate_id()
        logger.info(f"Ingesting PDF '{filename}' (id={document_id})")

        raw_docs = load_pdf(file_path)
        full_text = clean_text(" ".join(doc.page_content for doc in raw_docs))
        doc_metadata = extract_document_metadata(full_text, filename)
        doc_type = doc_metadata.get("document_type", "general")

        # ── 1. Text chunks ────────────────────────────────────────────────────
        child_chunks, parent_chunks = medical_chunk_documents(raw_docs)
        if not child_chunks:
            raise IngestionError(f"No text content extracted from '{filename}'")

        text_chunk_count = self._ingest_text_chunks(
            child_chunks, parent_chunks, document_id, filename, doc_type, doc_metadata
        )

        # ── 2. Table chunks ───────────────────────────────────────────────────
        extracted_images, extracted_tables = self.pdf_extractor.extract(file_path)
        table_chunk_count = self._ingest_table_chunks(
            extracted_tables, document_id, filename, doc_type, doc_metadata
        )

        # ── 3. Image chunks ───────────────────────────────────────────────────
        image_chunk_count = self._ingest_image_chunks(
            extracted_images, document_id, filename, doc_type, doc_metadata
        )

        doc = DocumentModel(
            id=document_id,
            filename=filename,
            source=file_path,
            file_type="pdf",
            document_type=doc_type,
            upload_date=now_utc(),
            checksum=checksum,
            chunk_count=text_chunk_count + table_chunk_count + image_chunk_count,
            parent_chunk_count=len(parent_chunks),
            metadata={
                **doc_metadata,
                "text_chunks": text_chunk_count,
                "table_chunks": table_chunk_count,
                "image_chunks": image_chunk_count,
                "images_extracted": len(extracted_images),
                "tables_extracted": len(extracted_tables),
            },
        )
        self.document_store.save(doc)

        logger.info(
            f"PDF ingestion complete: '{filename}' → "
            f"text={text_chunk_count}, tables={table_chunk_count}, images={image_chunk_count} chunks"
        )
        return doc

    def _ingest_text_chunks(
        self,
        child_chunks: list[dict],
        parent_chunks: list[dict],
        document_id: str,
        filename: str,
        doc_type: str,
        doc_metadata: dict,
    ) -> int:
        child_texts = [c["text"] for c in child_chunks]
        self.bm25_encoder.update(child_texts)
        child_embeddings = self.embedding_service.embed_documents(child_texts)

        child_points = []
        for chunk_data, embedding in zip(child_chunks, child_embeddings):
            chunk_id = generate_id()
            sparse_indices, sparse_values = self.bm25_encoder.encode_document(chunk_data["text"])
            payload = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "parent_chunk_id": chunk_data["parent_chunk_id"],
                "filename": filename,
                "text": chunk_data["text"],
                "section": chunk_data.get("section", "general"),
                "document_type": doc_type,
                "chunk_type": CHUNK_TYPE_TEXT,
                "specialty": doc_metadata.get("specialty"),
                "year": doc_metadata.get("year"),
                "authors": doc_metadata.get("authors"),
                "journal": doc_metadata.get("journal"),
                **chunk_data.get("metadata", {}),
            }
            child_points.append(PointStruct(
                id=chunk_id,
                vector={
                    DENSE_VECTOR_NAME: embedding,
                    SPARSE_VECTOR_NAME: SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload=payload,
            ))
        self.qdrant_store.upsert_chunks(child_points)

        parent_texts = [p["text"] for p in parent_chunks]
        if parent_texts:
            parent_embeddings = self.embedding_service.embed_documents(parent_texts)
            parent_points = []
            for parent_data, embedding in zip(parent_chunks, parent_embeddings):
                parent_payload = {
                    "parent_chunk_id": parent_data["parent_chunk_id"],
                    "document_id": document_id,
                    "filename": filename,
                    "text": parent_data["text"],
                    "section": parent_data.get("section", "general"),
                    "document_type": doc_type,
                    "chunk_type": CHUNK_TYPE_TEXT,
                    "specialty": doc_metadata.get("specialty"),
                    "year": doc_metadata.get("year"),
                    **parent_data.get("metadata", {}),
                }
                parent_points.append(PointStruct(
                    id=parent_data["parent_chunk_id"],
                    vector=embedding,
                    payload=parent_payload,
                ))
            self.qdrant_store.upsert_parents(parent_points)

        return len(child_chunks)

    def _ingest_table_chunks(
        self,
        tables: list,
        document_id: str,
        filename: str,
        doc_type: str,
        doc_metadata: dict,
    ) -> int:
        if not tables:
            return 0

        table_texts = []
        for table in tables:
            table_text = (
                f"[TABLE from page {table.page_number}]\n"
                f"{table.markdown_text}"
            )
            table_texts.append(table_text)

        self.bm25_encoder.update(table_texts)
        embeddings = self.embedding_service.embed_documents(table_texts)

        points = []
        for table, text, embedding in zip(tables, table_texts, embeddings):
            chunk_id = generate_id()
            sparse_indices, sparse_values = self.bm25_encoder.encode_document(text)
            payload = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "text": text,
                "section": "table",
                "document_type": doc_type,
                "chunk_type": CHUNK_TYPE_TABLE,
                "page": table.page_number,
                "table_id": table.table_id,
                "table_index": table.table_index,
                "specialty": doc_metadata.get("specialty"),
                "year": doc_metadata.get("year"),
            }
            points.append(PointStruct(
                id=chunk_id,
                vector={
                    DENSE_VECTOR_NAME: embedding,
                    SPARSE_VECTOR_NAME: SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload=payload,
            ))

        self.qdrant_store.upsert_chunks(points)
        logger.info(f"Ingested {len(points)} table chunks from '{filename}'")
        return len(points)

    def _ingest_image_chunks(
        self,
        images: list,
        document_id: str,
        filename: str,
        doc_type: str,
        doc_metadata: dict,
    ) -> int:
        if not images:
            return 0

        points = []
        for extracted_img in images:
            try:
                image_model = self.image_service.analyze_pil_image(
                    pil_image=extracted_img.pil_image,
                    image_id=extracted_img.image_id,
                    original_filename=filename,
                    page_number=extracted_img.page_number,
                    context_text=extracted_img.near_text,
                    caption=extracted_img.caption,
                )

                chunk_text = self.image_service.get_image_as_text(image_model)
                self.bm25_encoder.update([chunk_text])
                embedding = self.embedding_service.embed_query(chunk_text)
                sparse_indices, sparse_values = self.bm25_encoder.encode_document(chunk_text)
                chunk_id = generate_id()

                payload = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "text": chunk_text,
                    "section": "imaging",
                    "document_type": doc_type,
                    "chunk_type": CHUNK_TYPE_IMAGE,
                    "page": extracted_img.page_number,
                    "image_id": extracted_img.image_id,
                    "image_index": extracted_img.image_index,
                    "modality": image_model.modality,
                    "image_width": image_model.width,
                    "image_height": image_model.height,
                    "overall_severity": image_model.report.overall_severity,
                    "overall_confidence": image_model.report.overall_confidence,
                    "findings_count": len(image_model.findings),
                    "is_image_chunk": True,
                    "critical_flags": image_model.report.critical_flags,
                    "report_impression": image_model.report.impression,
                    "caption": image_model.caption or "",
                    "ocr_text": image_model.ocr_text or "",
                    "specialty": doc_metadata.get("specialty"),
                    "year": doc_metadata.get("year"),
                }

                points.append(PointStruct(
                    id=chunk_id,
                    vector={
                        DENSE_VECTOR_NAME: embedding,
                        SPARSE_VECTOR_NAME: SparseVector(indices=sparse_indices, values=sparse_values),
                    },
                    payload=payload,
                ))

                logger.debug(
                    f"Image chunk created: page={extracted_img.page_number} "
                    f"modality={image_model.modality} findings={len(image_model.findings)}"
                )

            except Exception as e:
                logger.warning(
                    f"Image analysis failed for image {extracted_img.image_id} "
                    f"page={extracted_img.page_number}: {e}"
                )
                continue

        if points:
            self.qdrant_store.upsert_chunks(points)
            logger.info(f"Ingested {len(points)} image chunks from '{filename}'")

        return len(points)

    # ── TXT / MD Pipeline ─────────────────────────────────────────────────────

    def _process_text_docs(
        self, raw_docs, file_path: str, filename: str, file_type: str, checksum: str
    ) -> DocumentModel:
        document_id = generate_id()
        logger.info(f"Ingesting '{filename}' (id={document_id})")

        full_text = clean_text(" ".join(doc.page_content for doc in raw_docs))
        doc_metadata = extract_document_metadata(full_text, filename)
        doc_type = doc_metadata.get("document_type", "general")

        child_chunks, parent_chunks = medical_chunk_documents(raw_docs)
        if not child_chunks:
            raise IngestionError(f"No content extracted from '{filename}'")

        text_count = self._ingest_text_chunks(
            child_chunks, parent_chunks, document_id, filename, doc_type, doc_metadata
        )

        doc = DocumentModel(
            id=document_id,
            filename=filename,
            source=file_path,
            file_type=file_type,
            document_type=doc_type,
            upload_date=now_utc(),
            checksum=checksum,
            chunk_count=text_count,
            parent_chunk_count=len(parent_chunks),
            metadata=doc_metadata,
        )
        self.document_store.save(doc)

        logger.info(f"Ingestion complete: '{filename}' → {text_count} text chunks, type={doc_type}")
        return doc
