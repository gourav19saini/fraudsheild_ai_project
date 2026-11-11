"""
FraudShield AI - Model Training
==================================
Trains an unsupervised Isolation Forest on engineered behavioral
features from data/transactions.csv and saves the fitted model +
scaler to models/isolation_forest.joblib.

Run:
    python -m backend.ml.train
or:
    python backend/ml/train.py
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# allow running as a plain script (python backend/ml/train.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.features import engineer_features_batch, FEATURE_COLUMNS  # noqa: E402

DATA_PATH = os.path.join("data", "transactions.csv")
MODEL_PATH = os.path.join("models", "isolation_forest.joblib")
CONTAMINATION = 0.04  # expected fraction of anomalies, tuned close to real fraud rate


def train_model(data_path: str = DATA_PATH, model_path: str = MODEL_PATH, contamination: float = CONTAMINATION):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Run `python data/generate_dataset.py` first."
        )

    print(f"[train] Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)

    print(f"[train] Engineering features for {len(df)} transactions ...")
    t0 = time.time()
    feat_df = engineer_features_batch(df)
    print(f"[train] Feature engineering done in {time.time() - t0:.2f}s")

    X = feat_df[FEATURE_COLUMNS].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[train] Training IsolationForest (contamination={contamination}) ...")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    train_time = time.time() - t0
    print(f"[train] Training completed in {train_time:.2f}s")

    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_rate = float((predictions == -1).mean())
    print(f"[train] Model-detected anomaly rate on training data: {anomaly_rate * 100:.2f}%")

    if "fraud_label" in feat_df.columns:
        true_rate = feat_df["fraud_label"].mean()
        print(f"[train] Ground-truth (informational only) fraud rate in dataset: {true_rate * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(df)),
        "anomaly_rate": anomaly_rate,
        "contamination": contamination,
        "version": "v1",
    }

    # If a previous model exists, bump the version number
    if os.path.exists(model_path):
        try:
            old_bundle = joblib.load(model_path)
            old_version = old_bundle.get("version", "v1")
            old_num = int(old_version.replace("v", "")) if old_version.replace("v", "").isdigit() else 1
            bundle["version"] = f"v{old_num + 1}"
        except Exception:
            bundle["version"] = "v1"

    joblib.dump(bundle, model_path)
    print(f"[train] Model saved to {model_path} (version {bundle['version']})")

    metadata = {
        "model_name": "IsolationForest",
        "version": bundle["version"],
        "trained_at": bundle["trained_at"],
        "training_samples": bundle["training_samples"],
        "anomaly_rate": bundle["anomaly_rate"],
        "training_time_seconds": round(train_time, 2),
    }
    return metadata


if __name__ == "__main__":
    meta = train_model()
    print(json.dumps(meta, indent=2))
