"""
FraudShield AI - Model Evaluation
====================================
Because the Isolation Forest is trained unsupervised, we use the
`fraud_label` column in the synthetic dataset ONLY for evaluation
(never for training) to report precision, recall, F1, confusion
matrix and ROC-AUC / PR-AUC.

IMPORTANT: With a highly imbalanced dataset (~3-5% fraud), accuracy
is a misleading metric - a model that predicts "SAFE" for everything
would still score >95% accuracy while catching zero fraud. Precision,
recall, F1 and ROC-AUC are reported instead because they account for
the class imbalance.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score,
)

from backend.ml.features import engineer_features_batch, FEATURE_COLUMNS

DATA_PATH = os.path.join("data", "transactions.csv")
MODEL_PATH = os.path.join("models", "isolation_forest.joblib")


def evaluate_model(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Train it first.")

    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    df = pd.read_csv(data_path)
    feat_df = engineer_features_batch(df)

    X = feat_df[FEATURE_COLUMNS].fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)

    raw_pred = model.predict(X_scaled)  # -1 anomaly, 1 normal
    pred_binary = (raw_pred == -1).astype(int)
    anomaly_scores = -model.decision_function(X_scaled)  # higher = more anomalous

    y_true = feat_df["fraud_label"].astype(int)

    cm = confusion_matrix(y_true, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, pred_binary)), 4),
        "precision": round(float(precision_score(y_true, pred_binary, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred_binary, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, pred_binary, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, anomaly_scores)), 4) if y_true.nunique() > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, anomaly_scores)), 4) if y_true.nunique() > 1 else None,
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "total_samples": int(len(y_true)),
        "actual_fraud_count": int(y_true.sum()),
        "predicted_anomaly_count": int(pred_binary.sum()),
        "note": (
            "Dataset is highly imbalanced (~3-5% fraud). Accuracy alone is misleading here - "
            "a model predicting 'safe' for everything would score >95% accuracy while catching "
            "zero fraud. Precision, recall, F1 and ROC-AUC better reflect real detection quality."
        ),
    }
    return metrics


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate_model(), indent=2))
