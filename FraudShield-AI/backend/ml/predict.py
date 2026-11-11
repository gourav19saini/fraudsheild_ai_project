"""
FraudShield AI - Prediction Engine
=====================================
Loads the trained Isolation Forest bundle and produces anomaly
predictions + scores for individual transactions in real time.
"""

from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
import joblib

from backend.ml.features import (
    UserHistoryStore,
    engineer_features_for_transaction,
    features_to_vector,
    FEATURE_COLUMNS,
)

MODEL_PATH = os.path.join("models", "isolation_forest.joblib")


class FraudPredictor:
    """
    Wraps the trained model + scaler and exposes a single `predict()`
    method used by the FastAPI layer. Keeps a shared UserHistoryStore
    so real-time features (frequency, new device, etc.) stay accurate
    across the life of the running backend process.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = FEATURE_COLUMNS
        self.metadata = {}
        self.history_store = UserHistoryStore()
        self.load()

    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self):
        if os.path.exists(self.model_path):
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
            self.metadata = {
                "version": bundle.get("version", "v1"),
                "trained_at": bundle.get("trained_at"),
                "training_samples": bundle.get("training_samples"),
                "anomaly_rate": bundle.get("anomaly_rate"),
                "contamination": bundle.get("contamination"),
            }
        else:
            self.model = None
            self.scaler = None
            self.metadata = {}

    def reload(self):
        """Re-read the model file from disk (used after retraining)."""
        self.load()

    def seed_history(self, transactions: list[dict]):
        """Preload the in-memory history store from past DB transactions
        so real-time features are accurate right after a backend restart."""
        for txn in sorted(transactions, key=lambda t: t["timestamp"]):
            self.history_store.add_transaction(txn["user_id"], txn)

    def predict(self, txn: dict) -> dict:
        """
        txn must contain: user_id, amount, merchant, merchant_category,
        location, latitude, longitude, device_id, timestamp (datetime).

        Returns a dict with features, raw_score, is_anomaly, latency_ms.
        """
        start = time.perf_counter()

        if not self.is_loaded():
            raise RuntimeError(
                "Model is not loaded. Train it first with `python backend/ml/train.py` "
                "or call POST /api/v1/model/retrain."
            )

        features = engineer_features_for_transaction(txn, self.history_store)
        vector_df = pd.DataFrame([features_to_vector(features)], columns=self.feature_columns)
        vector_scaled = self.scaler.transform(vector_df)

        raw_score = float(self.model.decision_function(vector_scaled)[0])
        raw_pred = int(self.model.predict(vector_scaled)[0])  # -1 anomaly, 1 normal
        is_anomaly = raw_pred == -1

        # commit this transaction into history AFTER scoring so features
        # reflect state "as of just before" this transaction, like real life
        self.history_store.add_transaction(txn["user_id"], txn)

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "features": features,
            "raw_score": raw_score,
            "is_anomaly": is_anomaly,
            "latency_ms": round(latency_ms, 3),
        }


# module-level singleton, imported by the FastAPI app
predictor = FraudPredictor()
