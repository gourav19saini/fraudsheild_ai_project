@echo off
REM FraudShield AI - Backend Launcher (Windows)
REM Run this from the project root (FraudShield-AI folder).

echo ==========================================
echo   FraudShield AI - Starting Backend
echo ==========================================

IF NOT EXIST venv\Scripts\activate.bat (
    echo [!] Virtual environment not found. Create one first:
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

IF NOT EXIST data\transactions.csv (
    echo [*] No dataset found - generating synthetic transactions...
    python data\generate_dataset.py
)

IF NOT EXIST models\isolation_forest.joblib (
    echo [*] No trained model found - training Isolation Forest...
    python backend\ml\train.py
)

echo [*] Starting FastAPI backend on http://localhost:8000 ...
uvicorn backend.main:app --reload --port 8000

pause
