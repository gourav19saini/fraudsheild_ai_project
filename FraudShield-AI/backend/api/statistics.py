"""
FraudShield AI - Statistics API
===================================
GET /api/v1/statistics - aggregate KPIs computed from real stored data.
"""

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database.database import get_db
from backend.database import models
from backend.schemas.transaction import StatisticsOut

router = APIRouter(prefix="/api/v1", tags=["statistics"])


@router.get("/statistics", response_model=StatisticsOut)
def get_statistics(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Transaction.id)).scalar() or 0

    if total == 0:
        return StatisticsOut(
            total_transactions=0, safe_count=0, review_count=0, high_risk_count=0,
            average_risk_score=0.0, anomaly_rate_pct=0.0,
            average_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
        )

    safe = db.query(func.count(models.Transaction.id)).filter(models.Transaction.risk_level == "SAFE").scalar() or 0
    review = db.query(func.count(models.Transaction.id)).filter(models.Transaction.risk_level == "REVIEW").scalar() or 0
    high = db.query(func.count(models.Transaction.id)).filter(models.Transaction.risk_level == "HIGH RISK").scalar() or 0
    avg_score = db.query(func.avg(models.Transaction.risk_score)).scalar() or 0.0
    anomalies = db.query(func.count(models.Transaction.id)).filter(models.Transaction.anomaly == True).scalar() or 0  # noqa: E712

    latencies = [row[0] for row in db.query(models.Transaction.prediction_latency_ms).all() if row[0] is not None]
    if latencies:
        avg_latency = float(np.mean(latencies))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99)) if len(latencies) >= 10 else float(np.max(latencies))
    else:
        avg_latency = p95 = p99 = 0.0

    return StatisticsOut(
        total_transactions=total,
        safe_count=safe,
        review_count=review,
        high_risk_count=high,
        average_risk_score=round(float(avg_score), 2),
        anomaly_rate_pct=round((anomalies / total) * 100, 2),
        average_latency_ms=round(avg_latency, 3),
        p95_latency_ms=round(p95, 3),
        p99_latency_ms=round(p99, 3),
    )
