"""
FraudShield AI - Live Stream Controller
==========================================
Runs a background thread that continuously POSTs generated transactions
to the FastAPI backend, simulating a real-time transaction feed. Used
by the Streamlit dashboard's "START LIVE STREAM" / "STOP LIVE STREAM"
controls.
"""

import threading
import time
import random
import requests

from simulator.generator import generate_transaction


class TransactionStream:
    def __init__(self, api_base_url: str = "http://localhost:8000", interval_seconds: float = 2.0):
        self.api_base_url = api_base_url
        self.interval_seconds = interval_seconds
        self._thread = None
        self._stop_event = threading.Event()
        self.is_running = False
        self.sent_count = 0
        self.last_error = None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                # 90% normal, 10% suspicious to keep the live feed realistic
                kind = "normal" if random.random() < 0.9 else "suspicious"
                txn = generate_transaction(kind)
                resp = requests.post(f"{self.api_base_url}/api/v1/transactions", json=txn, timeout=5)
                if resp.status_code == 200:
                    self.sent_count += 1
                    self.last_error = None
                else:
                    self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                self.last_error = str(e)
            self._stop_event.wait(self.interval_seconds)

    def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.is_running = True

    def stop(self):
        self._stop_event.set()
        self.is_running = False


# module-level singleton reused across Streamlit reruns via st.session_state
_stream_instance = None


def get_stream(api_base_url: str = "http://localhost:8000") -> TransactionStream:
    global _stream_instance
    if _stream_instance is None:
        _stream_instance = TransactionStream(api_base_url=api_base_url)
    return _stream_instance
