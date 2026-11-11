"""
FraudShield AI - Feature Engineering
=====================================
Turns raw transaction fields into numerical behavioral features used by
the Isolation Forest model and the risk engine.

This module is intentionally free of any external API calls - all
geo-distance computation is done locally with the Haversine formula.
"""

from __future__ import annotations
import math
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Columns fed into the Isolation Forest model, in this exact order.
FEATURE_COLUMNS = [
    "amount",
    "amount_ratio",
    "amount_zscore",
    "transactions_last_5_minutes",
    "transactions_last_1_hour",
    "time_since_previous_transaction",
    "distance_from_previous_km",
    "geo_velocity_kmh",
    "new_device",
    "new_merchant",
    "merchant_frequency",
    "location_frequency",
    "hour_of_day",
    "day_of_week",
    "is_night",
]


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in kilometers."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [lat1, lon1, lat2, lon2]):
        return 0.0
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


class UserHistoryStore:
    """
    Keeps a rolling in-memory history of each user's transactions so that
    real-time features (frequency, new device/merchant, running average)
    can be computed for streaming/live predictions, matching how the
    batch feature-engineering (used for training) computes them.
    """

    def __init__(self):
        self.history: dict[str, list[dict]] = {}

    def get_user_history(self, user_id: str) -> list[dict]:
        return self.history.get(user_id, [])

    def add_transaction(self, user_id: str, txn: dict):
        self.history.setdefault(user_id, []).append(txn)
        # keep history bounded for memory safety
        if len(self.history[user_id]) > 500:
            self.history[user_id] = self.history[user_id][-500:]

    def user_seen_device(self, user_id: str, device_id: str) -> bool:
        return any(t.get("device_id") == device_id for t in self.get_user_history(user_id))

    def user_seen_merchant(self, user_id: str, merchant: str) -> bool:
        return any(t.get("merchant") == merchant for t in self.get_user_history(user_id))

    def user_avg_amount(self, user_id: str) -> tuple[float, float]:
        hist = self.get_user_history(user_id)
        if not hist:
            return 0.0, 0.0
        amounts = [t["amount"] for t in hist]
        return float(np.mean(amounts)), float(np.std(amounts)) if len(amounts) > 1 else 1.0

    def transactions_in_window(self, user_id: str, now: datetime, minutes: int) -> int:
        hist = self.get_user_history(user_id)
        count = 0
        for t in hist:
            ts = t["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            delta = (now - ts).total_seconds() / 60.0
            if 0 <= delta <= minutes:
                count += 1
        return count

    def merchant_frequency(self, user_id: str, merchant: str) -> float:
        hist = self.get_user_history(user_id)
        if not hist:
            return 0.0
        matches = sum(1 for t in hist if t.get("merchant") == merchant)
        return matches / len(hist)

    def location_frequency(self, user_id: str, location: str) -> float:
        hist = self.get_user_history(user_id)
        if not hist:
            return 0.0
        matches = sum(1 for t in hist if t.get("location") == location)
        return matches / len(hist)

    def last_transaction(self, user_id: str) -> Optional[dict]:
        hist = self.get_user_history(user_id)
        return hist[-1] if hist else None


def engineer_features_for_transaction(txn: dict, store: "UserHistoryStore") -> dict:
    """
    Compute the full feature vector for a SINGLE incoming transaction using
    the rolling history stored in `store`. Used for real-time /predict calls.
    `txn` must contain: user_id, amount, merchant, merchant_category,
    location, latitude, longitude, device_id, timestamp (datetime).
    """
    user_id = txn["user_id"]
    now = txn["timestamp"]
    if isinstance(now, str):
        now = pd.to_datetime(now)

    is_first_ever_transaction = len(store.get_user_history(user_id)) == 0
    avg_amount, std_amount = store.user_avg_amount(user_id)
    std_amount = std_amount if std_amount > 1e-6 else 1.0
    amount = float(txn["amount"])

    amount_ratio = amount / avg_amount if avg_amount > 0 else 1.0
    amount_zscore = (amount - avg_amount) / std_amount if avg_amount > 0 else 0.0

    txns_5min = store.transactions_in_window(user_id, now, 5)
    txns_1hr = store.transactions_in_window(user_id, now, 60)

    last_txn = store.last_transaction(user_id)
    if last_txn is not None:
        last_ts = last_txn["timestamp"]
        if isinstance(last_ts, str):
            last_ts = pd.to_datetime(last_ts)
        time_since_prev = max(0.0, (now - last_ts).total_seconds() / 60.0)  # minutes
        distance_km = haversine_distance(
            last_txn.get("latitude"), last_txn.get("longitude"),
            txn.get("latitude"), txn.get("longitude"),
        )
        hours_elapsed = max(time_since_prev / 60.0, 1e-6)
        geo_velocity = distance_km / hours_elapsed
    else:
        time_since_prev = 999.0
        distance_km = 0.0
        geo_velocity = 0.0

    if is_first_ever_transaction:
        # Nothing to compare against yet - don't penalize a user's very
        # first observed transaction for being "new" on every dimension.
        new_device = 0
        new_merchant = 0
        merchant_freq = 1.0
        location_freq = 1.0
        amount_ratio = 1.0
        amount_zscore = 0.0
    else:
        new_device = 0 if store.user_seen_device(user_id, txn.get("device_id")) else 1
        new_merchant = 0 if store.user_seen_merchant(user_id, txn.get("merchant")) else 1
        merchant_freq = store.merchant_frequency(user_id, txn.get("merchant"))
        location_freq = store.location_frequency(user_id, txn.get("location"))

    hour_of_day = now.hour
    day_of_week = now.weekday()
    is_night = 1 if (hour_of_day <= 5 or hour_of_day >= 23) else 0

    features = {
        "amount": amount,
        "amount_ratio": round(amount_ratio, 4),
        "amount_zscore": round(amount_zscore, 4),
        "transactions_last_5_minutes": txns_5min,
        "transactions_last_1_hour": txns_1hr,
        "time_since_previous_transaction": round(time_since_prev, 2),
        "distance_from_previous_km": round(distance_km, 2),
        "geo_velocity_kmh": round(geo_velocity, 2),
        "new_device": new_device,
        "new_merchant": new_merchant,
        "merchant_frequency": round(merchant_freq, 4),
        "location_frequency": round(location_freq, 4),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_night": is_night,
    }
    return features


def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized-ish batch feature engineering used for TRAINING on the
    historical transactions.csv dataset. Processes transactions in
    chronological order per user, replaying history exactly like the
    real-time engine would see it.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    store = UserHistoryStore()
    feature_rows = []

    for _, row in df.iterrows():
        txn = {
            "user_id": row["user_id"],
            "amount": row["amount"],
            "merchant": row["merchant"],
            "merchant_category": row.get("merchant_category"),
            "location": row["location"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "device_id": row["device_id"],
            "timestamp": row["timestamp"],
        }
        feats = engineer_features_for_transaction(txn, store)
        feature_rows.append(feats)
        store.add_transaction(row["user_id"], txn)

    feat_df = pd.DataFrame(feature_rows)
    # "amount" is already present in the raw df with the same value -
    # drop the duplicate to avoid ambiguous/duplicate columns downstream.
    feat_df = feat_df.drop(columns=["amount"])
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return result


def features_to_vector(features: dict) -> list:
    """Order a feature dict into the exact vector expected by the model."""
    return [features[col] for col in FEATURE_COLUMNS]
