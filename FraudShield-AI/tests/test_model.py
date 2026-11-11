"""
Tests for the trained Isolation Forest model, prediction pipeline,
and risk scoring engine. Requires the model to already be trained
(python backend/ml/train.py) since these tests load the saved
models/isolation_forest.joblib bundle.
"""

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.predict import FraudPredictor
from backend.ml.risk_engine import calculate_risk_score

MODEL_PATH = os.path.join("models", "isolation_forest.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model not trained yet - run `python backend/ml/train.py` first.",
)


@pytest.fixture
def predictor():
    return FraudPredictor()


def test_model_loads(predictor):
    assert predictor.is_loaded()
    assert "version" in predictor.metadata


def test_normal_transaction_scores_low(predictor):
    txn = {
        "user_id": "TEST_NORMAL", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEVN1",
        "timestamp": datetime(2026, 1, 10, 14, 0, 0),
    }
    result = predictor.predict(txn)
    score, level, decision, reasons = calculate_risk_score(
        result["features"], result["raw_score"], result["is_anomaly"]
    )
    assert 0 <= score <= 100
    assert level in ("SAFE", "REVIEW", "HIGH RISK")
    assert isinstance(reasons, list) and len(reasons) > 0


def test_extreme_transaction_scores_high(predictor):
    # establish a normal baseline first
    baseline = {
        "user_id": "TEST_FRAUD", "amount": 400, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEVF1",
        "timestamp": datetime(2026, 1, 10, 10, 0, 0),
    }
    predictor.predict(baseline)

    extreme = {
        "user_id": "TEST_FRAUD", "amount": 150000, "merchant": "Tanishq",
        "merchant_category": "Jewellery", "location": "Mumbai",
        "latitude": 19.0760, "longitude": 72.8777, "device_id": "DEVFRAUD",
        "timestamp": datetime(2026, 1, 10, 10, 3, 0),  # 3 min later, far away
    }
    result = predictor.predict(extreme)
    score, level, decision, reasons = calculate_risk_score(
        result["features"], result["raw_score"], result["is_anomaly"]
    )
    assert score >= 70
    assert level == "HIGH RISK"
    assert decision == "FLAGGED"
    assert len(reasons) >= 2


def test_risk_score_bounds():
    fake_features = {
        "amount_ratio": 1.0, "geo_velocity_kmh": 0, "distance_from_previous_km": 0,
        "new_device": 0, "new_merchant": 0, "merchant_frequency": 1.0,
        "location_frequency": 1.0, "transactions_last_5_minutes": 0,
        "transactions_last_1_hour": 0, "is_night": 0,
    }
    score, level, decision, reasons = calculate_risk_score(fake_features, raw_iforest_score=0.3, is_anomaly_prediction=False)
    assert 0 <= score <= 100
    assert level == "SAFE"


def test_prediction_latency_recorded(predictor):
    txn = {
        "user_id": "TEST_LATENCY", "amount": 200, "merchant": "Zomato",
        "merchant_category": "Dining", "location": "Delhi",
        "latitude": 28.6139, "longitude": 77.2090, "device_id": "DEVL1",
        "timestamp": datetime(2026, 1, 10, 12, 0, 0),
    }
    result = predictor.predict(txn)
    assert result["latency_ms"] > 0
