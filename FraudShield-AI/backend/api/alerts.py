"""
FraudShield AI - Alerts API
==============================
GET /api/v1/alerts - list the most recent high-risk / review alerts.
"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.database import models
from backend.schemas.transaction import AlertOut

router = APIRouter(prefix="/api/v1", tags=["alerts"])


@router.get("/alerts", response_model=List[AlertOut])
def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    status: Optional[str] = Query(None, description="Filter by OPEN / REVIEWED / DISMISSED"),
    db: Session = Depends(get_db),
):
    query = db.query(models.Alert)
    if status:
        query = query.filter(models.Alert.status == status.upper())
    rows = query.order_by(desc(models.Alert.created_at)).limit(limit).all()
    result = []
    for r in rows:
        reasons = json.loads(r.reasons) if r.reasons else []
        result.append(
            AlertOut(
                alert_id=r.alert_id,
                transaction_id=r.transaction_id,
                risk_score=r.risk_score,
                risk_level=r.risk_level,
                reasons=reasons,
                created_at=r.created_at,
                status=r.status,
            )
        )
    return result
