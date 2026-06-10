# Medical Multimodal RAG System

Production-grade Medical Retrieval-Augmented Generation system with full multimodal support, BGE + Cross-Encoder reranking, context validation, evidence verification, and calibrated confidence scoring.

## Features

### Retrieval Pipeline
```
Query → Query Enhancement (Multi-Query + HyDE) → Embedding (BGE-M3) →
Hybrid Retrieval (Dense + BM25/Sparse) → RRF Fusion → Parent Retrieval →
Multimodal Image Retrieval → Reranker (BGE + CrossEncoder + Medical Domain) →
Context Selection → Context Validation → Prompt Builder → LLM
```

### Answer Format
Every response returns:
- **Answer** — grounded, citation-marked medical answer
- **Sources** — numbered citations with metadata, rerank scores, image flags
- **Confidence Score** — calibrated breakdown (retrieval, grounding, coverage, hallucination risk)

### Medical Image Support
| Modality | Preprocessing | Analysis |
|----------|--------------|----------|
| X-Ray | CLAHE, denoising, normalization | Opacity, density, consolidation detection |
| CT | CLAHE, normalization | Hyperintense/hypointense regions, HU analysis |
| MRI | Denoising, normalization | Signal characteristics, lesion detection |
| Ultrasound | Contrast enhancement | Echogenicity distribution, fluid detection |
| Pathology | Contrast enhancement | H&E staining analysis, nuclear morphology |
| Clinical | Contrast enhancement | Lesion characterization, wound assessment |
| DICOM | Full metadata extraction | Patient/study/series metadata |

### Reranking Pipeline
- **BGE Reranker** (`BAAI/bge-reranker-base`) — semantic cross-attention scoring
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — relevance scoring
- **Medical Domain Reranker** — section/type/terminology boosts
- **Score Fusion** — weighted combination of all rerankers

### Validation & Calibration
- **Context Validation** — quality, coverage, consistency scoring
- **Evidence Verification** — claim extraction + source support checking
- **Unsupported Claim Detection** — flags numeric/definitive claims lacking context support
- **Final Confidence Calibration** — Bayesian-style fusion of all signals

## Project Structure

```
medical_rag_output/
├── src/
│   ├── controllers/
│   │   ├── BaseController.py
│   │   └── RAGController.py              # Orchestrates full pipeline
│   ├── services/
│   │   ├── EmbeddingService.py           # BGE-M3 dense embeddings
│   │   ├── GenerationService.py          # LLM generation + validation
│   │   ├── IngestionService.py           # Document ingestion
│   │   ├── MedicalImageService.py        # Image ingestion + analysis
│   │   ├── MultimodalRetrievalService.py # Combined text+image retrieval
│   │   ├── QueryEnhancementService.py    # Multi-Query + HyDE
│   │   ├── RerankingService.py           # BGE + CrossEncoder + Medical
│   │   ├── RetrievalService.py           # Hybrid retrieval pipeline
│   │   └── ContextValidationService.py  # Validation + calibration
│   ├── medical/
│   │   ├── bm25_encoder.py
│   │   ├── citation_generator.py
│   │   ├── confidence_scorer.py
│   │   ├── metadata_extractor.py
│   │   └── section_splitter.py
│   ├── models/
│   │   ├── ChunkModel.py
│   │   ├── DocumentModel.py
│   │   ├── MedicalImageModel.py          # Image + findings + report models
│   │   └── schemas.py                    # API request/response schemas
│   ├── stores/
│   │   ├── DocumentStore.py
│   │   └── QdrantStore.py
│   ├── routes/
│   │   ├── base.py
│   │   ├── dependencies.py
│   │   └── rag.py                        # REST endpoints
│   ├── langchain_components/
│   │   ├── loaders.py
│   │   └── prompts.py
│   ├── helpers/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── logger.py
│   │   └── utils.py
│   └── main.py
├── uploads/
│   └── images/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup & Run Instructions

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Tesseract OCR (for OCR support)
sudo apt-get install tesseract-ocr      # Ubuntu/Debian
brew install tesseract                   # macOS

# Qdrant (via Docker)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum
```

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run

```bash
cd medical_rag_output
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### 5. Test

```bash
pytest tests/ -v
```

## API Endpoints

### POST /api/v1/ingest
Ingest a medical document (PDF, TXT, MD).

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@patient_record.pdf"
```

### POST /api/v1/ingest/image
Ingest a medical image (X-Ray, CT, MRI, etc.).

```bash
curl -X POST http://localhost:8000/api/v1/ingest/image \
  -F "file=@chest_xray.jpg" \
  -F "modality=xray" \
  -F "clinical_context=Patient presenting with shortness of breath"
```

### POST /api/v1/query
Query the system. Returns Answer + Sources + Confidence.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the treatment options for type 2 diabetes?",
    "use_reranking": true,
    "use_multimodal": true,
    "use_hyde": true
  }'
```

**Response structure:**
```json
{
  "question": "...",
  "answer": "...[1][2]...\n\nReferences:\n[1] ...\n[2] ...",
  "sources": [
    {
      "citation_number": 1,
      "filename": "diabetes_guideline.pdf",
      "score": 87.4,
      "rerank_score": 0.923,
      "text_preview": "...",
      "is_image_source": false
    }
  ],
  "confidence": {
    "overall": 0.78,
    "retrieval_confidence": 0.82,
    "answer_grounding": 0.74,
    "source_coverage": 0.71,
    "hallucination_risk": 0.18,
    "is_reliable": true,
    "warnings": []
  },
  "context_validation": {
    "is_valid": true,
    "quality_score": 0.81,
    "coverage_score": 0.76,
    "consistency_score": 0.90
  },
  "evidence_verification": {
    "support_ratio": 0.85,
    "evidence_strength": "strong",
    "verified_claims_count": 6,
    "unsupported_claims_count": 1
  },
  "retrieval_strategy": "Query Enhancement → Embedding → Hybrid Retrieval → ... → LLM",
  "image_chunks_used": 1,
  "reranking_applied": true
}
```

### GET /api/v1/health
### GET /api/v1/stats

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `GROQ_MODEL` | `llama-3.1-70b-versatile` | LLM model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `BGE_RERANKER_MODEL` | `BAAI/bge-reranker-base` | BGE reranker |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `ENABLE_BGE_RERANKER` | `true` | Enable BGE reranking |
| `ENABLE_CROSS_ENCODER` | `true` | Enable cross-encoder reranking |
| `ENABLE_MULTIMODAL` | `true` | Enable image retrieval |
| `RERANK_TOP_N` | `5` | Chunks after reranking |
| `TOP_K` | `8` | Retrieval top-k |
| `CONFIDENCE_THRESHOLD` | `0.55` | Reliability threshold |

## Notes

- BGE reranker requires `FlagEmbedding`: `pip install FlagEmbedding`
- DICOM support requires `pydicom`: `pip install pydicom`
- VLM image analysis uses Groq's vision endpoint when a compatible model is configured
- Rerankers gracefully degrade if models unavailable (pass-through mode)
- All image modalities use pixel-level analysis as fallback when VLM is unavailable
