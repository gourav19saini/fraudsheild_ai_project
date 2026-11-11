# 🛡️ FraudShield AI
### Real-Time Fraud & Anomaly Detection System
**Problem Statement PS-02: Real-Time Algorithmic Fraud & Anomaly Detection in Streaming Data**

---

## 1. Problem Understanding

Financial fraud happens in milliseconds and adapts constantly. A usable fraud
detection system for streaming transactions must:
- Score every transaction in real time (low latency)
- Work without labeled fraud data (real fraud labels are rare/delayed)
- Explain *why* a transaction was flagged, not just output a number
- Detect when transaction patterns change (concept drift) and adapt

## 2. Solution

FraudShield AI is a self-contained, local, real-time transaction risk-scoring
platform. It ingests transactions (via API or the built-in live simulator),
engineers behavioral features (spending habits, device history, geo-velocity,
timing), scores each one with an **unsupervised Isolation Forest**, converts
that into an explainable 0-100 risk score, stores everything in SQLite, and
visualizes it all on a live Streamlit dashboard — with concept-drift
monitoring and one-click model retraining.

No external paid APIs. No cloud infrastructure. Runs entirely on a laptop.

## 3. Key Features

- Real-time transaction scoring via a FastAPI REST API
- Isolation Forest anomaly detection (unsupervised — never trained on the fraud label)
- Explainable, factor-based 0-100 risk scoring (not random, not a black box)
- Human-readable fraud reasons generated from actual feature values
- SQLite persistence for transactions, alerts, and model metadata
- Live transaction stream simulator (normal / suspicious / extreme fraud)
- Statistical concept-drift detector (no River dependency required)
- One-click model retraining from the dashboard
- Professional fintech-style Streamlit dashboard with live charts
- Precision / Recall / F1 / ROC-AUC evaluation against labeled synthetic data
- Full latency tracking (avg / p95 / p99)
- Pytest test suite for features, model, and API

## 4. Architecture

```
Transaction Generator (simulator/)
        |
        v
FastAPI Backend  (backend/main.py + backend/api/*)
        |
        v
Feature Engineering  (backend/ml/features.py)
        |
        v
Isolation Forest  (backend/ml/predict.py, backend/ml/train.py)
        |
        v
Risk Engine  (backend/ml/risk_engine.py)
        |
        v
Explanation Engine  (built into risk_engine.py)
        |
        v
SQLite  (backend/database/*)
        |
        v
Streamlit Dashboard  (frontend/dashboard.py)
```

## 5. Technology Stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| Language       | Python 3.11+                         |
| Backend        | FastAPI, Uvicorn, Pydantic           |
| ML             | Scikit-learn (Isolation Forest), Pandas, NumPy |
| Database       | SQLite + SQLAlchemy                  |
| Frontend       | Streamlit + Plotly                   |
| Utilities      | Joblib, python-dotenv                |
| Testing        | Pytest, httpx (FastAPI TestClient)   |

No Kafka, Redis, Docker, Kubernetes, PostgreSQL, React, or Node.js — kept
intentionally lean for a 6-hour hackathon build and a Windows/VS Code setup.

## 6. Machine Learning Methodology

### Why Isolation Forest?
Isolation Forest is an unsupervised anomaly detection algorithm that isolates
observations by randomly partitioning the feature space. Anomalies are
"few and different," so they get isolated in fewer random splits than normal
points — this gives an anomaly score without needing labeled fraud examples,
which matches real-world fraud detection where confirmed fraud labels are
scarce and delayed.

The synthetic dataset DOES include a `fraud_label` column, but it is used
**only for evaluation** (Section 9 below) — never fed into training. The
model learns purely from the shape of the feature distribution.

### Feature Engineering (`backend/ml/features.py`)
15 behavioral features are computed per transaction:
`amount, amount_ratio, amount_zscore, transactions_last_5_minutes,
transactions_last_1_hour, time_since_previous_transaction,
distance_from_previous_km, geo_velocity_kmh, new_device, new_merchant,
merchant_frequency, location_frequency, hour_of_day, day_of_week, is_night`

Geo-distance is computed locally using the **Haversine formula** (no Google
Maps / external geo API). A user's very first-ever transaction is treated as
a neutral baseline rather than penalized for "looking new on everything."

### Risk Scoring (`backend/ml/risk_engine.py`)
The Isolation Forest's raw anomaly score contributes up to 40 of the 100
points; the remaining 60 points come from concrete, auditable behavioral
signals (amount ratio, geo-velocity/impossible travel, new device, unusual
merchant/location, transaction burst frequency, odd-hour timing). Nothing is
random — every point is traceable to a specific feature value.

- **0-30 -> SAFE** (APPROVED)
- **31-70 -> REVIEW** (FLAGGED_FOR_REVIEW)
- **71-100 -> HIGH RISK** (FLAGGED)

### Explanation Engine
Reasons are generated directly from whichever risk factors actually fired
for that transaction (e.g. *"Transaction amount is 170.0x above the user's
normal average"*, *"Impossible travel detected: 921 km covered at ~18,415
km/h since last transaction"*). No LLM, no hard-coded templates repeated
across transactions, fully offline.

## 7. Concept Drift Detection (`backend/ml/drift.py`)

A lightweight, dependency-free statistical detector compares a rolling
"current window" of recent transaction amounts against a "reference window"
using a z-score on the mean plus a relative-change threshold:

- **NORMAL** — distributions are statistically similar
- **WARNING** — moderate shift (|z| >= 2 or relative change >= 60%)
- **DETECTED** — strong shift (|z| >= 4 or relative change >= 150%)

Click **SIMULATE CONCEPT DRIFT** on the dashboard to inject a burst of
transactions with a shifted amount distribution and watch the status change
live.

## 8. Model Retraining

`POST /api/v1/model/retrain` (or the dashboard's **RETRAIN MODEL** button)
reloads `data/transactions.csv`, re-engineers features, fits a fresh
Isolation Forest, saves it with an incremented version number
(`models/isolation_forest.joblib`), and hot-reloads it into the running
backend — no restart required.

## 9. Evaluation Metrics

Because only ~3.5% of transactions are fraudulent, **accuracy is misleading**
— a model that predicts "safe" for every transaction would score >95%
accuracy while catching zero fraud. FraudShield instead reports:

- Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC and Precision-Recall AUC

Measured on this project's synthetic dataset: **Precision ≈ 0.63, Recall ≈
0.71, F1 ≈ 0.67, ROC-AUC ≈ 0.95** (exact numbers vary slightly by dataset seed
and contamination setting — view live numbers under the dashboard's *Model &
Drift* tab or via `GET /api/v1/model/evaluation`).

## 10. Database Design

**transactions** — every scored transaction: raw fields + risk_score,
risk_level, decision, anomaly flag, reasons (JSON), prediction latency.

**alerts** — one row per REVIEW/HIGH RISK transaction, with status
(OPEN/REVIEWED/DISMISSED) for triage workflows.

**model_metadata** — model name, version, trained_at, training_samples,
anomaly_rate for each retrain.

## 11. API Documentation

Interactive Swagger UI is available at **http://localhost:8000/docs** once
the backend is running. Endpoint list:

| Method | Endpoint                        | Description |
|--------|----------------------------------|--------------|
| GET    | `/`                              | API info |
| GET    | `/health`                        | Health check |
| POST   | `/api/v1/transactions`           | Ingest + score + store a transaction |
| GET    | `/api/v1/transactions`           | List recent transactions |
| GET    | `/api/v1/transactions/{id}`      | Get one transaction |
| POST   | `/api/v1/predict`                | Score a transaction (same pipeline) |
| GET    | `/api/v1/alerts`                 | List REVIEW/HIGH RISK alerts |
| GET    | `/api/v1/statistics`             | KPI dashboard stats |
| GET    | `/api/v1/model/status`           | Model monitoring info |
| POST   | `/api/v1/model/retrain`          | Retrain the Isolation Forest |
| GET    | `/api/v1/model/evaluation`       | Precision/Recall/F1/ROC-AUC |
| GET    | `/api/v1/drift/status`           | Current concept-drift status |
| POST   | `/api/v1/drift/simulate`         | Inject a drift scenario (demo) |

### Example Request

```json
POST /api/v1/predict
{
  "user_id": "U1001",
  "amount": 85000,
  "merchant": "Electronics Store",
  "merchant_category": "Electronics",
  "location": "Delhi",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "device_id": "DEV999"
}
```

### Example Response

```json
{
  "transaction_id": "TXA1B2C3D4E5",
  "user_id": "U1001",
  "amount": 85000,
  "risk_score": 94,
  "risk_level": "HIGH RISK",
  "decision": "FLAGGED",
  "anomaly": true,
  "reasons": [
    "Isolation Forest model flagged this transaction pattern as statistically unusual",
    "Transaction amount is 35.2x above the user's normal average",
    "New device detected"
  ],
  "prediction_latency_ms": 12.4,
  "timestamp": "2026-09-25T10:15:00"
}
```

## 12. Installation (Windows + VS Code)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 13. Running Instructions

```bat
REM 1. Generate the synthetic dataset (12,000 transactions)
python data\generate_dataset.py

REM 2. Train the Isolation Forest model
python backend\ml\train.py

REM 3. Start the backend  (Terminal 1)
uvicorn backend.main:app --reload --port 8000
REM   ...or simply double-click / run:  run_backend.bat

REM 4. Start the dashboard  (Terminal 2, new window)
streamlit run frontend\dashboard.py
```

The backend auto-trains the model on first startup if `data/transactions.csv`
exists but `models/isolation_forest.joblib` doesn't — steps 1-2 above are a
safety net so everything works even the very first time.

Dashboard: **http://localhost:8501**
API Swagger docs: **http://localhost:8000/docs**

## 14. 2-Minute Judge Demo Procedure

1. Open the dashboard — point out the KPI cards (all real DB-backed numbers, all zero initially).
2. Sidebar -> click **START** to begin the live transaction stream.
3. Switch to the **Live Monitor** tab — watch transactions arrive and get scored in real time.
4. Sidebar -> click **Generate SUSPICIOUS transaction** — switch to **Alerts** tab, show the flagged transaction with its reasons.
5. Sidebar -> click **💀 Generate EXTREME FRAUD transaction** — show the 90+ HIGH RISK score and its detailed reasons (impossible travel, new device, huge amount ratio).
6. Sidebar -> click **SIMULATE CONCEPT DRIFT** — go to **Model & Drift** tab, show the status flip from NORMAL to WARNING/DETECTED.
7. Sidebar -> click **RETRAIN MODEL** — show the version bump and updated training stats.
8. Show the **Model & Drift** tab's evaluation metrics (Precision/Recall/F1/ROC-AUC + confusion matrix) to demonstrate this isn't a black box.

## 15. Running Tests

```bat
pytest tests/ -v
```

`test_features.py` and `test_model.py` test the ML core directly (no server
needed). `test_api.py` uses FastAPI's `TestClient` to exercise every endpoint
in-process.

## 16. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Activate the venv and re-run `pip install -r requirements.txt` |
| Dashboard shows "Backend OFFLINE" | Make sure `uvicorn backend.main:app --port 8000` is running in another terminal |
| `/predict` returns 503 "Model is not loaded" | Run `python backend\ml\train.py`, or call `POST /api/v1/model/retrain` |
| Empty charts / "No transaction data yet" | Start the live stream or click a "Generate ... transaction" button in the sidebar |
| `data/transactions.csv` missing | Run `python data\generate_dataset.py` |
| Port 8000 or 8501 already in use | `uvicorn backend.main:app --port 8001` and set `FRAUDSHIELD_API_URL=http://localhost:8001` in `.env` |

## 17. Limitations

- Synthetic data only — real transaction data would need domain-specific validation
- Isolation Forest is a strong general-purpose anomaly detector but not the
  final word; a production system would ensemble it with supervised models
  once labeled fraud feedback becomes available
- The drift detector monitors transaction *amount* distribution as the primary
  signal for the hackathon demo; a production version would monitor the full
  feature vector distribution
- SQLite is fine for a demo/hackathon; a production deployment would use a
  more concurrent database

## 18. Future Improvements

- Ensemble Isolation Forest with a supervised model (e.g. XGBoost) once
  labeled feedback loops are available
- Multi-feature (not just amount) concept-drift monitoring using
  population stability index (PSI) or ADWIN across the full feature vector
- Per-user adaptive thresholds instead of global risk bands
- Graph-based features (shared devices/merchants across users) for
  organized-fraud-ring detection
- WebSocket-based push updates to the dashboard instead of polling/rerun

## How This Solves PS-02

PS-02 asks for **real-time algorithmic fraud & anomaly detection in
streaming data**. FraudShield AI directly answers this: a live transaction
stream is continuously scored end-to-end (ingestion -> feature engineering ->
unsupervised ML -> explainable risk scoring -> storage -> live dashboard) in
milliseconds per transaction, with concept-drift monitoring and retraining to
handle the fact that streaming data distributions change over time — all
running locally with zero external paid API dependencies.
