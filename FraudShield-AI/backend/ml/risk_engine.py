"""
FraudShield AI - Risk Scoring Engine
======================================
Converts the Isolation Forest's raw anomaly score plus individual
behavioral signals into an explainable 0-100 risk score, a risk level,
and a decision. Also produces human-readable reasons.

Nothing here is random - every point added to the score is traceable to
a concrete feature value.
"""

from __future__ import annotations
from typing import Tuple, List


def _normalize_iforest_score(raw_score: float) -> float:
    """
    sklearn's IsolationForest.decision_function returns roughly
    [-0.5, 0.5], where LOWER (more negative) = more anomalous.
    We flip and rescale to a 0-100 "model anomaly score" where
    HIGHER = more anomalous.
    """
    clipped = max(-0.5, min(0.5, raw_score))
    normalized = (0.5 - clipped) / 1.0  # 0 (normal) .. 1 (very anomalous)
    return max(0.0, min(1.0, normalized)) * 100


def calculate_risk_score(features: dict, raw_iforest_score: float, is_anomaly_prediction: bool) -> Tuple[int, str, str, List[str]]:
    """
    Returns (risk_score:int, risk_level:str, decision:str, reasons:list[str]).

    Score composition (weights sum to 100 max, then clipped):
      - Base model anomaly score (Isolation Forest)      : up to 40 pts
      - Amount ratio vs personal average                  : up to 20 pts
      - Geo-velocity (impossible travel)                   : up to 15 pts
      - New device                                          : 10 pts
      - New / unusual merchant                              : 5 pts
      - Unusual location (low location_frequency)           : 5 pts
      - High transaction frequency (burst)                  : up to 10 pts
      - Odd hour (night-time transaction)                    : 5 pts
    """
    reasons = []
    score = 0.0

    # 1. Model-driven base score
    model_score = _normalize_iforest_score(raw_iforest_score)
    model_contribution = (model_score / 100.0) * 40.0
    score += model_contribution
    if model_score > 60:
        reasons.append("Isolation Forest model flagged this transaction pattern as statistically unusual")

    # 2. Amount ratio vs personal average
    amount_ratio = features.get("amount_ratio", 1.0)
    if amount_ratio >= 30:
        score += 20
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x above the user's normal average")
    elif amount_ratio >= 10:
        score += 15
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x above the user's normal average")
    elif amount_ratio >= 4:
        score += 9
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x higher than usual for this user")
    elif amount_ratio >= 2.5:
        score += 4
        reasons.append("Transaction amount is moderately above the user's typical spending")

    # 3. Geo-velocity / impossible travel
    geo_velocity = features.get("geo_velocity_kmh", 0.0)
    distance_km = features.get("distance_from_previous_km", 0.0)
    if geo_velocity >= 800 and distance_km > 50:
        score += 15
        reasons.append(f"Impossible travel detected: {distance_km:.0f} km covered at ~{geo_velocity:.0f} km/h since last transaction")
    elif geo_velocity >= 200 and distance_km > 50:
        score += 8
        reasons.append(f"Unusually high geo-velocity ({geo_velocity:.0f} km/h) between consecutive transactions")

    # 4. New device
    if features.get("new_device"):
        score += 10
        reasons.append("Transaction made from a device never used before by this user")

    # 5. New / unusual merchant
    if features.get("new_merchant") and features.get("merchant_frequency", 0) < 0.05:
        score += 5
        reasons.append("Merchant is unusual for this user's typical spending pattern")

    # 6. Unusual location
    if features.get("location_frequency", 1.0) < 0.1:
        score += 5
        reasons.append("Transaction location is unusual for this user")

    # 7. Burst / high frequency
    txns_5min = features.get("transactions_last_5_minutes", 0)
    txns_1hr = features.get("transactions_last_1_hour", 0)
    if txns_5min >= 3:
        score += 10
        reasons.append(f"High transaction velocity: {txns_5min} transactions in the last 5 minutes")
    elif txns_1hr >= 5:
        score += 6
        reasons.append(f"Elevated transaction frequency: {txns_1hr} transactions in the last hour")

    # 8. Odd hour
    if features.get("is_night"):
        score += 5
        reasons.append("Transaction occurred at an unusual hour (late night)")

    # If the model itself is confident this is an outlier, nudge the score up further
    if is_anomaly_prediction:
        score += 5

    score = int(round(max(0, min(100, score))))

    if score <= 30:
        risk_level = "SAFE"
        decision = "APPROVED"
    elif score <= 70:
        risk_level = "REVIEW"
        decision = "FLAGGED_FOR_REVIEW"
    else:
        risk_level = "HIGH RISK"
        decision = "FLAGGED"

    if not reasons:
        reasons.append("No significant anomaly signals detected; transaction matches user's normal behavior")

    return score, risk_level, decision, reasons
