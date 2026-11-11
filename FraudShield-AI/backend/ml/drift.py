"""
FraudShield AI - Concept Drift Detection
===========================================
A lightweight, dependency-free statistical drift detector. Compares a
recent "current window" of transaction amounts (and anomaly rate)
against a "reference window" established during/soon-after training,
using a z-test style comparison of means plus a relative-change
threshold. This avoids the need for the `river` library so the
hackathon project stays install-friendly on Windows.

Status levels:
    NORMAL   - distributions are statistically similar
    WARNING  - moderate shift detected
    DETECTED - strong shift detected (concept drift)
"""

from __future__ import annotations
import math
from collections import deque
from typing import Optional


class DriftDetector:
    def __init__(self, reference_window_size: int = 200, current_window_size: int = 50):
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.reference_amounts: deque = deque(maxlen=reference_window_size)
        self.current_amounts: deque = deque(maxlen=current_window_size)
        self.reference_established = False
        self.last_status = "NORMAL"
        self.last_details = {}

    def seed_reference(self, amounts: list[float]):
        """Populate the reference window (e.g. from training data)."""
        for a in amounts[-self.reference_window_size:]:
            self.reference_amounts.append(float(a))
        self.reference_established = len(self.reference_amounts) >= 20

    def add_observation(self, amount: float):
        self.current_amounts.append(float(amount))
        if not self.reference_established:
            # bootstrap reference window from the earliest live data too
            if len(self.reference_amounts) < self.reference_window_size:
                self.reference_amounts.append(float(amount))
            self.reference_established = len(self.reference_amounts) >= 20

    def _mean_std(self, data):
        n = len(data)
        if n == 0:
            return 0.0, 0.0
        mean = sum(data) / n
        var = sum((x - mean) ** 2 for x in data) / n if n > 1 else 0.0
        return mean, math.sqrt(var)

    def check_drift(self) -> dict:
        """
        Returns a dict describing current drift status. Uses a simple
        Welch-style z-score on the mean transaction amount between the
        reference window and the current (recent) window.
        """
        if len(self.current_amounts) < 5 or not self.reference_established:
            self.last_status = "NORMAL"
            self.last_details = {
                "status": "NORMAL",
                "message": "Not enough data yet to evaluate drift.",
                "reference_mean": None,
                "current_mean": None,
                "z_score": None,
            }
            return self.last_details

        ref_mean, ref_std = self._mean_std(list(self.reference_amounts))
        cur_mean, cur_std = self._mean_std(list(self.current_amounts))

        ref_std = ref_std if ref_std > 1e-6 else 1.0
        n_cur = len(self.current_amounts)
        standard_error = ref_std / math.sqrt(n_cur)
        z_score = (cur_mean - ref_mean) / standard_error if standard_error > 0 else 0.0

        relative_change = abs(cur_mean - ref_mean) / ref_mean if ref_mean > 0 else 0.0

        if abs(z_score) >= 4 or relative_change >= 1.5:
            status = "DETECTED"
            message = f"Concept drift detected: average amount shifted from ~₹{ref_mean:,.0f} to ~₹{cur_mean:,.0f}"
        elif abs(z_score) >= 2 or relative_change >= 0.6:
            status = "WARNING"
            message = f"Possible drift: average amount moved from ~₹{ref_mean:,.0f} to ~₹{cur_mean:,.0f}"
        else:
            status = "NORMAL"
            message = "Transaction pattern is stable, no significant drift."

        self.last_status = status
        self.last_details = {
            "status": status,
            "message": message,
            "reference_mean": round(ref_mean, 2),
            "current_mean": round(cur_mean, 2),
            "z_score": round(z_score, 2),
            "relative_change_pct": round(relative_change * 100, 1),
        }
        return self.last_details

    def reset_current_window(self):
        self.current_amounts.clear()


# module-level singleton shared across the FastAPI app
drift_detector = DriftDetector()
