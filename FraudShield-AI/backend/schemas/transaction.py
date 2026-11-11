"""
FraudShield AI - Pydantic Schemas
====================================
Request/response validation models for the FastAPI layer.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class TransactionIn(BaseModel):
    """Incoming transaction payload for POST /api/v1/transactions and /predict."""
    user_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, description="Transaction amount, must be > 0")
    merchant: str = Field(..., min_length=1)
    merchant_category: Optional[str] = "Other"
    location: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    device_id: str = Field(..., min_length=1)
    timestamp: Optional[datetime] = None

    @field_validator("user_id", "merchant", "location", "device_id")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field cannot be blank")
        return v.strip()


class PredictionOut(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    risk_score: int
    risk_level: str
    decision: str
    anomaly: bool
    reasons: List[str]
    prediction_latency_ms: float
    timestamp: datetime


class TransactionOut(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    merchant_category: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    device_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None
    decision: Optional[str] = None
    anomaly: Optional[bool] = None
    reasons: Optional[List[str]] = None
    prediction_latency_ms: Optional[float] = None

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    alert_id: str
    transaction_id: str
    risk_score: int
    risk_level: str
    reasons: List[str]
    created_at: Optional[datetime] = None
    status: str

    class Config:
        from_attributes = True


class StatisticsOut(BaseModel):
    total_transactions: int
    safe_count: int
    review_count: int
    high_risk_count: int
    average_risk_score: float
    anomaly_rate_pct: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class ModelStatusOut(BaseModel):
    model_name: str
    status: str
    version: str
    transactions_processed: int
    anomalies_detected: int
    anomaly_rate_pct: float
    drift_status: str
    last_retrained: Optional[str] = None
    training_samples: Optional[int] = None


class RetrainOut(BaseModel):
    message: str
    version: str
    training_samples: int
    training_time_seconds: float
    anomaly_rate_pct: float


class DriftStatusOut(BaseModel):
    status: str
    message: str
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    z_score: Optional[float] = None
    relative_change_pct: Optional[float] = None
