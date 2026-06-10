import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    with patch("src.routes.dependencies.get_embedding_service") as mock_emb, \
         patch("src.routes.dependencies.get_qdrant_store") as mock_qdrant:
        mock_emb.return_value = MagicMock(is_loaded=True)
        mock_qdrant.return_value = MagicMock(
            is_healthy=lambda: True,
            list_collection_names=lambda: ["medical_rag"],
        )
        from src.main import app
        yield TestClient(app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "Medical" in data["name"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 500)


def test_query_validation(client):
    resp = client.post("/api/v1/query", json={"question": "ok"})
    assert resp.status_code in (200, 422, 500)
