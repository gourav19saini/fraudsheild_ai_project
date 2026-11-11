"""
Tests for the FastAPI backend endpoints. Uses FastAPI's TestClient,
which runs the app in-process (no need to start uvicorn separately).

Run:
    pytest tests/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "FraudShield AI"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_valid_transaction():
    payload = {
        "user_id": "API_TEST_USER",
        "amount": 750,
        "merchant": "BigBasket",
        "merchant_category": "Grocery",
        "location": "Jaipur",
        "latitude": 26.9124,
        "longitude": 75.7873,
        "device_id": "DEV_API_1",
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "risk_score" in body
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in ("SAFE", "REVIEW", "HIGH RISK")


def test_predict_rejects_invalid_amount():
    payload = {
        "user_id": "API_TEST_USER", "amount": -50, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEV_API_1",
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422  # pydantic validation error


def test_predict_rejects_invalid_latitude():
    payload = {
        "user_id": "API_TEST_USER", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 500, "longitude": 75.7873, "device_id": "DEV_API_1",
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_list_transactions():
    resp = client.get("/api/v1/transactions?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_nonexistent_transaction_returns_404():
    resp = client.get("/api/v1/transactions/TX_DOES_NOT_EXIST")
    assert resp.status_code == 404


def test_statistics_endpoint():
    resp = client.get("/api/v1/statistics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_transactions" in body
    assert "average_latency_ms" in body


def test_alerts_endpoint():
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_model_status_endpoint():
    resp = client.get("/api/v1/model/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "IsolationForest"


def test_drift_status_endpoint():
    resp = client.get("/api/v1/drift/status")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_drift_simulate_endpoint():
    resp = client.post("/api/v1/drift/simulate")
    assert resp.status_code == 200
    assert "drift_result" in resp.json()
