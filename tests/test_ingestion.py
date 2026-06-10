import pytest
from unittest.mock import MagicMock, patch
from src.services.IngestionService import IngestionService
from src.medical.bm25_encoder import BM25Encoder
from src.helpers.exceptions import UnsupportedFileTypeError


@pytest.fixture
def mock_services():
    embedding_service = MagicMock()
    embedding_service.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    qdrant_store = MagicMock()
    document_store = MagicMock()
    document_store.exists_by_checksum.return_value = None
    bm25_encoder = BM25Encoder()
    return embedding_service, qdrant_store, document_store, bm25_encoder


def test_unsupported_type_raises(mock_services):
    svc = IngestionService(*mock_services)
    with pytest.raises(UnsupportedFileTypeError):
        svc.process_document("/tmp/file.xyz", "file.xyz")


def test_bm25_encode_query():
    encoder = BM25Encoder()
    corpus = [
        "The patient presented with chest pain and dyspnea.",
        "Treatment with aspirin and beta-blockers was initiated.",
        "Laboratory results showed elevated troponin levels.",
    ]
    encoder.fit(corpus)
    indices, values = encoder.encode_query("chest pain treatment")
    assert len(indices) > 0
    assert len(indices) == len(values)
    assert all(v > 0 for v in values)


def test_bm25_fallback_without_fit():
    encoder = BM25Encoder()
    indices, values = encoder.encode_query("some medical query")
    assert isinstance(indices, list)
    assert isinstance(values, list)
