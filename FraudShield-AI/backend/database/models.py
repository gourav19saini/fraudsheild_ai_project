"""
FraudShield AI - Database Models
===================================
SQLAlchemy ORM models: Transaction, Alert, ModelMetadata.
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.sql import func
from backend.database.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=False)
    merchant_category = Column(String, nullable=True)
    location = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    device_id = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())

    risk_score = Column(Integer, nullable=True)
    risk_level = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    anomaly = Column(Boolean, default=False)
    reasons = Column(Text, nullable=True)  # JSON-encoded list of strings
    prediction_latency_ms = Column(Float, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String, unique=True, index=True, nullable=False)
    transaction_id = Column(String, index=True, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False)
    reasons = Column(Text, nullable=True)  # JSON-encoded list
    created_at = Column(DateTime, server_default=func.now())
    status = Column(String, default="OPEN")  # OPEN, REVIEWED, DISMISSED


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    trained_at = Column(DateTime, nullable=True)
    training_samples = Column(Integer, nullable=True)
    anomaly_rate = Column(Float, nullable=True)
