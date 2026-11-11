"""
FraudShield AI - Synthetic Transaction Dataset Generator
==========================================================
Generates a realistic dataset of financial transactions with a small
percentage of anomalous / fraudulent transactions mixed in.

Run:
    python data/generate_dataset.py
"""

import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_USERS = 250
N_TRANSACTIONS = 12000
FRAUD_RATE = 0.035  # ~3.5% anomalous transactions

MERCHANT_CATEGORIES = {
    "Grocery": ["BigBasket", "Reliance Fresh", "DMart", "Local Kirana"],
    "Electronics": ["Croma", "Reliance Digital", "Vijay Sales"],
    "Fashion": ["Zara", "H&M", "Myntra Store"],
    "Dining": ["Zomato", "Swiggy", "Domino's", "Local Cafe"],
    "Fuel": ["Indian Oil", "HP Petrol Pump", "Shell"],
    "Travel": ["MakeMyTrip", "IRCTC", "Uber", "Ola"],
    "Entertainment": ["BookMyShow", "PVR Cinemas", "Netflix"],
    "Utilities": ["Electricity Board", "Airtel", "Jio Recharge"],
    "Jewellery": ["Tanishq", "Kalyan Jewellers"],
    "Online": ["Amazon", "Flipkart", "Myntra"],
}

CITIES = {
    "Jaipur": (26.9124, 75.7873),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow": (26.8467, 80.9462),
}

CITY_NAMES = list(CITIES.keys())


def jitter_coords(lat, lon, km=3):
    """Add small random jitter (in km) to a lat/lon pair."""
    delta_deg = km / 111.0
    return lat + random.uniform(-delta_deg, delta_deg), lon + random.uniform(-delta_deg, delta_deg)


def make_user_profile(user_id):
    home_city = random.choice(CITY_NAMES)
    avg_amount = np.random.choice([300, 800, 1500, 3000, 6000], p=[0.35, 0.3, 0.2, 0.1, 0.05])
    avg_amount = float(avg_amount * np.random.uniform(0.7, 1.3))
    n_devices = random.choice([1, 1, 1, 2])
    devices = [f"DEV{random.randint(10000, 99999)}" for _ in range(n_devices)]
    fav_categories = random.sample(list(MERCHANT_CATEGORIES.keys()), k=random.randint(2, 4))
    return {
        "user_id": user_id,
        "home_city": home_city,
        "avg_amount": avg_amount,
        "std_amount": avg_amount * 0.35,
        "devices": devices,
        "fav_categories": fav_categories,
    }


def random_timestamp(start, end):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def generate_normal_transaction(profile, prev_txn, ts):
    category = random.choice(profile["fav_categories"])
    merchant = random.choice(MERCHANT_CATEGORIES[category])
    lat, lon = CITIES[profile["home_city"]]
    lat, lon = jitter_coords(lat, lon, km=5)
    amount = max(20.0, np.random.normal(profile["avg_amount"], profile["std_amount"] * 0.5))
    device = random.choice(profile["devices"])

    return {
        "amount": round(amount, 2),
        "merchant": merchant,
        "merchant_category": category,
        "location": profile["home_city"],
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "device_id": device,
        "timestamp": ts,
        "fraud_label": 0,
    }


def generate_anomalous_transaction(profile, prev_txn, ts):
    """Create a suspicious transaction using a random mixture of signals."""
    signals = random.sample(
        ["huge_amount", "new_device", "new_merchant", "new_location", "impossible_travel", "odd_hour", "burst"],
        k=random.randint(1, 3),
    )

    category = random.choice(list(MERCHANT_CATEGORIES.keys()))
    merchant = random.choice(MERCHANT_CATEGORIES[category])
    home_lat, home_lon = CITIES[profile["home_city"]]
    lat, lon = jitter_coords(home_lat, home_lon, km=5)
    location = profile["home_city"]
    amount = max(20.0, np.random.normal(profile["avg_amount"], profile["std_amount"] * 0.5))
    device = random.choice(profile["devices"])

    if "huge_amount" in signals:
        amount = profile["avg_amount"] * random.uniform(8, 40)

    if "new_device" in signals:
        device = f"DEV{random.randint(10000, 99999)}9"  # never-before-seen device

    if "new_merchant" in signals:
        category = random.choice([c for c in MERCHANT_CATEGORIES if c not in profile["fav_categories"]])
        merchant = random.choice(MERCHANT_CATEGORIES[category])

    if "new_location" in signals or "impossible_travel" in signals:
        far_city = random.choice([c for c in CITY_NAMES if c != profile["home_city"]])
        lat, lon = CITIES[far_city]
        lat, lon = jitter_coords(lat, lon, km=5)
        location = far_city

    if "odd_hour" in signals:
        ts = ts.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59))

    # impossible_travel: force this txn to be seconds/minutes after prev txn but far away
    if "impossible_travel" in signals and prev_txn is not None:
        ts = prev_txn["timestamp"] + timedelta(minutes=random.randint(1, 8))

    return {
        "amount": round(amount, 2),
        "merchant": merchant,
        "merchant_category": category,
        "location": location,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "device_id": device,
        "timestamp": ts,
        "fraud_label": 1,
    }


def generate_dataset(n_transactions=N_TRANSACTIONS, n_users=N_USERS, fraud_rate=FRAUD_RATE):
    profiles = {f"U{1000 + i}": make_user_profile(f"U{1000 + i}") for i in range(n_users)}
    end_time = datetime.now()
    start_time = end_time - timedelta(days=45)

    rows = []
    last_txn_by_user = {}

    # simulate a chronological stream, sampling users repeatedly
    timestamps = sorted(random_timestamp(start_time, end_time) for _ in range(n_transactions))

    for i, ts in enumerate(timestamps):
        user_id = random.choice(list(profiles.keys()))
        profile = profiles[user_id]
        prev_txn = last_txn_by_user.get(user_id)

        is_fraud = random.random() < fraud_rate
        if is_fraud:
            txn = generate_anomalous_transaction(profile, prev_txn, ts)
        else:
            txn = generate_normal_transaction(profile, prev_txn, ts)

        txn["transaction_id"] = f"TX{100000 + i}"
        txn["user_id"] = user_id

        if prev_txn is not None:
            txn["previous_transaction_time"] = prev_txn["timestamp"]
            txn["previous_latitude"] = prev_txn["latitude"]
            txn["previous_longitude"] = prev_txn["longitude"]
            txn["previous_amount"] = prev_txn["amount"]
        else:
            txn["previous_transaction_time"] = None
            txn["previous_latitude"] = None
            txn["previous_longitude"] = None
            txn["previous_amount"] = None

        rows.append(txn)
        last_txn_by_user[user_id] = txn

    df = pd.DataFrame(rows)
    cols = [
        "transaction_id", "user_id", "amount", "merchant", "merchant_category",
        "location", "latitude", "longitude", "device_id", "timestamp",
        "previous_transaction_time", "previous_latitude", "previous_longitude",
        "previous_amount", "fraud_label",
    ]
    df = df[cols].sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"Fraud rate: {df['fraud_label'].mean() * 100:.2f}%")
    print(df.head())
