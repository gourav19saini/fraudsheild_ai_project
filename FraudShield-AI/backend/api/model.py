"""
FraudShield AI - Model & Drift API
=====================================
GET  /api/v1/model/status
POST /api/v1/model/retrain
GET  /api/v1/model/evaluation   (extra: precision/recall/F1/ROC-AUC on synthetic labeled data)
GET  /api/v1/drift/status
POST /api/v1/drift/simulate
"""

import random
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database.database import get_db
from backend.database import models
from backend.schemas.transaction import ModelStatusOut, RetrainOut, DriftStatusOut
from backend.ml.predict import predictor
from backend.ml.drift import drift_detector
from backend.ml.train import train_model
from backend.ml.evaluate import evaluate_model

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model/status", response_model=ModelStatusOut)
def model_status(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Transaction.id)).scalar() or 0
    anomalies = db.query(func.count(models.Transaction.id)).filter(models.Transaction.anomaly == True).scalar() or 0  # noqa: E712
    anomaly_rate = (anomalies / total * 100) if total else 0.0

    drift_status = drift_detector.last_status

    if not predictor.is_loaded():
        return ModelStatusOut(
            model_name="IsolationForest",
            status="NOT_LOADED",
            version="-",
            transactions_processed=total,
            anomalies_detected=anomalies,
            anomaly_rate_pct=round(anomaly_rate, 2),
            drift_status=drift_status,
            last_retrained=None,
            training_samples=None,
        )

    meta = predictor.metadata
    return ModelStatusOut(
        model_name="IsolationForest",
        status="ACTIVE",
        version=meta.get("version", "v1"),
        transactions_processed=total,
        anomalies_detected=anomalies,
        anomaly_rate_pct=round(anomaly_rate, 2),
        drift_status=drift_status,
        last_retrained=meta.get("trained_at"),
        training_samples=meta.get("training_samples"),
    )


@router.post("/model/retrain", response_model=RetrainOut)
def retrain_model():
    try:
        meta = train_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    predictor.reload()
    drift_detector.reset_current_window()

    return RetrainOut(
        message="Model retrained successfully.",
        version=meta["version"],
        training_samples=meta["training_samples"],
        training_time_seconds=meta["training_time_seconds"],
        anomaly_rate_pct=round(meta["anomaly_rate"] * 100, 2),
    )


@router.get("/model/evaluation")
def model_evaluation():
    """Evaluate the current model against the labeled synthetic dataset."""
    try:
        return evaluate_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drift/status", response_model=DriftStatusOut)
def drift_status():
    details = drift_detector.check_drift()
    return DriftStatusOut(**details)


@router.post("/drift/simulate")
def simulate_drift():
    """
    Injects a burst of synthetic transactions with a shifted amount
    distribution into the drift detector's current window, to
    demonstrate drift detection live for the demo.
    """
    shift_multiplier = random.uniform(4, 8)
    base = drift_detector.reference_amounts[-1] if drift_detector.reference_amounts else 1000.0
    injected = []
    for _ in range(25):
        amt = max(50.0, random.gauss(base * shift_multiplier, base * 0.5))
        drift_detector.add_observation(amt)
        injected.append(round(amt, 2))

    details = drift_detector.check_drift()
    return {
        "message": "Simulated a shifted transaction pattern and re-evaluated drift.",
        "injected_sample_amounts": injected[:5],
        "drift_result": details,
    }
