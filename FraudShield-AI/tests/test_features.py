"""
Tests for backend/ml/features.py - haversine distance, cold-start
handling, and feature vector shape/ordering.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.features import (
    haversine_distance,
    UserHistoryStore,
    engineer_features_for_transaction,
    features_to_vector,
    FEATURE_COLUMNS,
)


def test_haversine_zero_distance():
    d = haversine_distance(26.9124, 75.7873, 26.9124, 75.7873)
    assert d == 0.0


def test_haversine_known_distance_jaipur_delhi():
    # Jaipur to Delhi is roughly 240-280 km as the crow flies
    d = haversine_distance(26.9124, 75.7873, 28.6139, 77.2090)
    assert 200 < d < 320


def test_haversine_handles_missing_values():
    d = haversine_distance(None, None, 28.6139, 77.2090)
    assert d == 0.0


def test_cold_start_first_transaction_not_penalized():
    store = UserHistoryStore()
    txn = {
        "user_id": "NEWUSER", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEV1",
        "timestamp": datetime.now(),
    }
    feats = engineer_features_for_transaction(txn, store)
    assert feats["new_device"] == 0
    assert feats["new_merchant"] == 0
    assert feats["amount_ratio"] == 1.0


def test_second_transaction_detects_new_device():
    store = UserHistoryStore()
    now = datetime.now()
    txn1 = {
        "user_id": "U1", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEVOLD",
        "timestamp": now,
    }
    engineer_features_for_transaction(txn1, store)
    store.add_transaction("U1", txn1)

    txn2 = dict(txn1)
    txn2["device_id"] = "DEVNEW"
    txn2["timestamp"] = now + timedelta(minutes=10)
    feats2 = engineer_features_for_transaction(txn2, store)
    assert feats2["new_device"] == 1


def test_feature_vector_matches_columns_order():
    store = UserHistoryStore()
    txn = {
        "user_id": "U1", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEV1",
        "timestamp": datetime.now(),
    }
    feats = engineer_features_for_transaction(txn, store)
    vector = features_to_vector(feats)
    assert len(vector) == len(FEATURE_COLUMNS)
    for col, val in zip(FEATURE_COLUMNS, vector):
        assert feats[col] == val


def test_impossible_travel_high_velocity():
    store = UserHistoryStore()
    now = datetime.now()
    txn1 = {
        "user_id": "U2", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Jaipur",
        "latitude": 26.9124, "longitude": 75.7873, "device_id": "DEV1",
        "timestamp": now,
    }
    engineer_features_for_transaction(txn1, store)
    store.add_transaction("U2", txn1)

    # 3 minutes later, transaction in Mumbai (~900+ km away)
    txn2 = {
        "user_id": "U2", "amount": 500, "merchant": "BigBasket",
        "merchant_category": "Grocery", "location": "Mumbai",
        "latitude": 19.0760, "longitude": 72.8777, "device_id": "DEV1",
        "timestamp": now + timedelta(minutes=3),
    }
    feats2 = engineer_features_for_transaction(txn2, store)
    assert feats2["geo_velocity_kmh"] > 500
