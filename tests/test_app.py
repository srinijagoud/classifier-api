"""
Basic tests for the sentiment classifier API.
Run: pytest tests/
"""

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely wonderful, I loved every minute."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "This was the worst film I have ever seen, a complete waste of time."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_missing_text_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # FastAPI validation error


def test_predict_empty_string():
    response = client.post("/predict", json={"text": ""})
    # Should still return 200 — model will just make some (possibly low-confidence) prediction
    assert response.status_code == 200