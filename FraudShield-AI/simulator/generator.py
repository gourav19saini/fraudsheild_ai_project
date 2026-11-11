"""
FraudShield AI - Live Transaction Generator
==============================================
Generates single transactions on demand for the streaming simulator:
normal, suspicious, or extreme-fraud patterns. Reuses the same city /
merchant catalogs as the offline dataset generator for consistency.
"""

import random
from datetime import datetime

CITIES = {
    "Jaipur": (26.9124, 75.7873),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
}

MERCHANTS = {
    "Grocery": ["BigBasket", "DMart", "Local Kirana"],
    "Electronics": ["Croma", "Reliance Digital", "Vijay Sales"],
    "Fashion": ["Zara", "H&M", "Myntra Store"],
    "Dining": ["Zomato", "Swiggy", "Domino's"],
    "Fuel": ["Indian Oil", "HP Petrol Pump"],
    "Travel": ["MakeMyTrip", "IRCTC", "Uber"],
    "Entertainment": ["BookMyShow", "PVR Cinemas"],
    "Jewellery": ["Tanishq", "Kalyan Jewellers"],
    "Online": ["Amazon", "Flipkart"],
}

# a small fixed pool of demo users/devices so the model can build history on them
DEMO_USERS = [f"U{2000 + i}" for i in range(30)]
DEMO_DEVICES = {u: f"DEV{random.randint(10000, 99999)}" for u in DEMO_USERS}


def _jitter(lat, lon, km=4):
    d = km / 111.0
    return lat + random.uniform(-d, d), lon + random.uniform(-d, d)


def generate_transaction(kind: str = "normal") -> dict:
    """
    kind: "normal" | "suspicious" | "extreme"
    Returns a plain dict matching the TransactionIn schema fields.
    """
    user_id = random.choice(DEMO_USERS)
    category = random.choice(list(MERCHANTS.keys()))
    merchant = random.choice(MERCHANTS[category])
    city = random.choice(list(CITIES.keys()))
    lat, lon = CITIES[city]
    lat, lon = _jitter(lat, lon)
    device_id = DEMO_DEVICES[user_id]

    if kind == "normal":
        amount = round(random.uniform(100, 3000), 2)

    elif kind == "suspicious":
        amount = round(random.uniform(15000, 45000), 2)
        # occasionally swap in a new device or a far city to add signal variety
        if random.random() < 0.5:
            device_id = f"DEV{random.randint(10000, 99999)}X"
        if random.random() < 0.5:
            far_city = random.choice([c for c in CITIES if c != city])
            lat, lon = CITIES[far_city]
            lat, lon = _jitter(lat, lon)
            city = far_city

    elif kind == "extreme":
        amount = round(random.uniform(60000, 250000), 2)
        device_id = f"DEV{random.randint(10000, 99999)}X"
        far_city = random.choice(list(CITIES.keys()))
        lat, lon = CITIES[far_city]
        lat, lon = _jitter(lat, lon)
        city = far_city
        category = random.choice(["Electronics", "Jewellery"])
        merchant = random.choice(MERCHANTS[category])

    else:
        raise ValueError(f"Unknown transaction kind: {kind}")

    return {
        "user_id": user_id,
        "amount": amount,
        "merchant": merchant,
        "merchant_category": category,
        "location": city,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
