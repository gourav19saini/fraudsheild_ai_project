"""
FraudShield AI - API Client
==============================
Thin wrapper around `requests` used by the Streamlit dashboard to talk
to the FastAPI backend. Every function fails gracefully (returns None /
empty structures + an error message) so the dashboard never crashes if
the backend is unreachable.
"""

import os
import requests

API_BASE_URL = os.getenv("FRAUDSHIELD_API_URL", "http://localhost:8000")
TIMEOUT = 8


def _get(path, params=None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API returned {resp.status_code}: {resp.text[:300]}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the FraudShield backend. Is it running? (uvicorn backend.main:app --port 8000)"
    except Exception as e:
        return None, str(e)


def _post(path, json_body=None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"API returned {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the FraudShield backend. Is it running? (uvicorn backend.main:app --port 8000)"
    except Exception as e:
        return None, str(e)


def check_health():
    return _get("/health")


def get_statistics():
    return _get("/api/v1/statistics")


def get_transactions(limit=50, risk_level=None):
    params = {"limit": limit}
    if risk_level:
        params["risk_level"] = risk_level
    return _get("/api/v1/transactions", params=params)


def get_alerts(limit=20):
    return _get("/api/v1/alerts", params={"limit": limit})


def get_model_status():
    return _get("/api/v1/model/status")


def get_model_evaluation():
    return _get("/api/v1/model/evaluation")


def get_drift_status():
    return _get("/api/v1/drift/status")


def simulate_drift():
    return _post("/api/v1/drift/simulate")


def retrain_model():
    return _post("/api/v1/model/retrain")


def analyze_transaction(payload: dict):
    return _post("/api/v1/predict", json_body=payload)


def submit_transaction(payload: dict):
    return _post("/api/v1/transactions", json_body=payload)
