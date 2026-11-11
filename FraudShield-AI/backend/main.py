"""
FraudShield AI - FastAPI Backend Entry Point
================================================
Run:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.database.database import init_db, SessionLocal
from backend.database import models
from backend.ml.predict import predictor
from backend.ml.drift import drift_detector
from backend.ml.train import train_model
from backend.api import transactions, alerts, statistics, model as model_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraudshield")


def _ensure_model_trained():
    """Auto-train the model on startup if no model file exists yet."""
    model_path = os.path.join("models", "isolation_forest.joblib")
    if not os.path.exists(model_path):
        dataset_path = os.path.join("data", "transactions.csv")
        if not os.path.exists(dataset_path):
            logger.warning(
                "No dataset found at data/transactions.csv - cannot auto-train. "
                "Run `python data/generate_dataset.py` then `python backend/ml/train.py`."
            )
            return
        logger.info("No trained model found - auto-training Isolation Forest now ...")
        train_model()
        predictor.reload()
        logger.info("Auto-training complete.")


def _seed_state_from_db():
    """Warm up the predictor's history store and drift detector reference
    window from whatever transactions already exist in the database, so
    a backend restart doesn't lose real-time context."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.Transaction)
            .order_by(models.Transaction.timestamp.asc())
            .limit(2000)
            .all()
        )
        history = [
            {
                "user_id": r.user_id, "amount": r.amount, "merchant": r.merchant,
                "merchant_category": r.merchant_category, "location": r.location,
                "latitude": r.latitude, "longitude": r.longitude,
                "device_id": r.device_id, "timestamp": r.timestamp,
            }
            for r in rows
        ]
        if history:
            predictor.seed_history(history)
            drift_detector.seed_reference([h["amount"] for h in history])
            logger.info(f"Seeded predictor + drift detector from {len(history)} stored transactions.")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _ensure_model_trained()
    _seed_state_from_db()
    if not drift_detector.reference_established and predictor.metadata:
        # bootstrap a reasonable drift reference so status isn't blank pre-traffic
        pass
    logger.info("FraudShield AI backend started.")
    yield
    logger.info("FraudShield AI backend shutting down.")


app = FastAPI(
    title="FraudShield AI",
    description="Real-Time Algorithmic Fraud & Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transactions.router)
app.include_router(alerts.router)
app.include_router(statistics.router)
app.include_router(model_router.router)


@app.get("/")
def root():
    return {
        "name": "FraudShield AI",
        "description": "Real-Time Algorithmic Fraud & Anomaly Detection System",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
        "model_version": predictor.metadata.get("version") if predictor.metadata else None,
    }
