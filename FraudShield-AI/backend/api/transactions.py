"""
FraudShield AI - Transactions & Prediction API
=================================================
POST /api/v1/transactions        - store a raw transaction (auto-scored)
GET  /api/v1/transactions         - list recent transactions
GET  /api/v1/transactions/{id}    - get a single transaction
POST /api/v1/predict              - score a transaction without necessarily
                                     persisting extra state beyond storage
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.database import models
from backend.schemas.transaction import TransactionIn, PredictionOut, TransactionOut
from backend.ml.predict import predictor
from backend.ml.risk_engine import calculate_risk_score
from backend.ml.drift import drift_detector

router = APIRouter(prefix="/api/v1", tags=["transactions"])


def _score_and_store(payload: TransactionIn, db: Session) -> dict:
    ts = payload.timestamp or datetime.utcnow()
    txn_id = f"TX{uuid.uuid4().hex[:10].upper()}"

    txn_dict = {
        "user_id": payload.user_id,
        "amount": payload.amount,
        "merchant": payload.merchant,
        "merchant_category": payload.merchant_category,
        "location": payload.location,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "device_id": payload.device_id,
        "timestamp": ts,
    }

    if not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded. Train it first (python backend/ml/train.py) "
                   "or call POST /api/v1/model/retrain.",
        )

    result = predictor.predict(txn_dict)
    score, level, decision, reasons = calculate_risk_score(
        result["features"], result["raw_score"], result["is_anomaly"]
    )

    # feed the drift detector with this transaction's amount
    drift_detector.add_observation(payload.amount)

    db_txn = models.Transaction(
        transaction_id=txn_id,
        user_id=payload.user_id,
        amount=payload.amount,
        merchant=payload.merchant,
        merchant_category=payload.merchant_category,
        location=payload.location,
        latitude=payload.latitude,
        longitude=payload.longitude,
        device_id=payload.device_id,
        timestamp=ts,
        risk_score=score,
        risk_level=level,
        decision=decision,
        anomaly=bool(result["is_anomaly"]),
        reasons=json.dumps(reasons),
        prediction_latency_ms=result["latency_ms"],
    )
    db.add(db_txn)
    db.commit()
    db.refresh(db_txn)

    if level in ("REVIEW", "HIGH RISK"):
        alert = models.Alert(
            alert_id=f"AL{uuid.uuid4().hex[:10].upper()}",
            transaction_id=txn_id,
            risk_score=score,
            risk_level=level,
            reasons=json.dumps(reasons),
            status="OPEN",
        )
        db.add(alert)
        db.commit()

    return {
        "transaction_id": txn_id,
        "user_id": payload.user_id,
        "amount": payload.amount,
        "risk_score": score,
        "risk_level": level,
        "decision": decision,
        "anomaly": bool(result["is_anomaly"]),
        "reasons": reasons,
        "prediction_latency_ms": result["latency_ms"],
        "timestamp": ts,
    }


@router.post("/predict", response_model=PredictionOut)
def predict_transaction(payload: TransactionIn, db: Session = Depends(get_db)):
    """Score a transaction in real time and persist the result."""
    return _score_and_store(payload, db)


@router.post("/transactions", response_model=PredictionOut)
def create_transaction(payload: TransactionIn, db: Session = Depends(get_db)):
    """Ingest + score + store a transaction (same pipeline as /predict)."""
    return _score_and_store(payload, db)


@router.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
    limit: int = Query(50, ge=1, le=1000),
    risk_level: Optional[str] = Query(None, description="Filter by SAFE / REVIEW / HIGH RISK"),
    db: Session = Depends(get_db),
):
    query = db.query(models.Transaction)
    if risk_level:
        query = query.filter(models.Transaction.risk_level == risk_level.upper())
    rows = query.order_by(desc(models.Transaction.timestamp)).limit(limit).all()
    return [_txn_to_out(r) for r in rows]


@router.get("/transactions/{transaction_id}", response_model=TransactionOut)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found")
    return _txn_to_out(row)


def _txn_to_out(row: models.Transaction) -> TransactionOut:
    reasons = json.loads(row.reasons) if row.reasons else []
    return TransactionOut(
        transaction_id=row.transaction_id,
        user_id=row.user_id,
        amount=row.amount,
        merchant=row.merchant,
        merchant_category=row.merchant_category,
        location=row.location,
        latitude=row.latitude,
        longitude=row.longitude,
        device_id=row.device_id,
        timestamp=row.timestamp,
        risk_score=row.risk_score,
        risk_level=row.risk_level,
        decision=row.decision,
        anomaly=row.anomaly,
        reasons=reasons,
        prediction_latency_ms=row.prediction_latency_ms,
    )
