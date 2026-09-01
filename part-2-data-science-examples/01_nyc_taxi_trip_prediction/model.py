"""Training and inference utilities for the taxi duration model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "taxi_duration.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "hour", "day_of_week",
    "is_weekend", "distance_km",
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def generate_sample(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Create a small, realistic NYC-shaped sample when Kaggle data is unavailable."""
    rng = np.random.default_rng(seed)
    pickup_lat = rng.normal(40.758, 0.035, rows).clip(40.62, 40.88)
    pickup_lon = rng.normal(-73.985, 0.045, rows).clip(-74.05, -73.75)
    dropoff_lat = (pickup_lat + rng.normal(0, 0.035, rows)).clip(40.58, 40.92)
    dropoff_lon = (pickup_lon + rng.normal(0, 0.045, rows)).clip(-74.10, -73.70)
    pickup = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 3600, rows), unit="s")
    distance = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    hour = pickup.hour.to_numpy()
    weekday = pickup.dayofweek.to_numpy()
    rush = ((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))
    speed = np.where(rush, 14.0, np.where((hour <= 5), 29.0, 21.0))
    # NYC streets make trips longer than straight-line distance; add signal and noise.
    seconds = 180 + (distance * 1.28 / speed * 3600) + rng.normal(0, 80, rows)
    seconds = np.clip(seconds, 90, 7200).round().astype(int)
    return pd.DataFrame({
        "id": [f"sample_{i}" for i in range(rows)],
        "pickup_datetime": pickup,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": rng.integers(1, 7, rows),
        "trip_duration": seconds,
    })


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    pickup = pd.to_datetime(data["pickup_datetime"])
    data["hour"] = pickup.dt.hour
    data["day_of_week"] = pickup.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["distance_km"] = haversine_km(
        data["pickup_latitude"], data["pickup_longitude"],
        data["dropoff_latitude"], data["dropoff_longitude"],
    )
    return data[FEATURES]


def load_training_data(csv_path: str | None = None, sample_rows: int = 6000) -> tuple[pd.DataFrame, str]:
    if csv_path:
        frame = pd.read_csv(csv_path, nrows=sample_rows, parse_dates=["pickup_datetime"])
        required = set(FEATURES[:5] + ["pickup_datetime", "trip_duration"])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        frame = frame.dropna(subset=list(required))
        frame = frame[frame["trip_duration"].between(60, 7200)]
        return frame, str(Path(csv_path).resolve())
    return generate_sample(sample_rows), "generated NYC-like sample (seed=42)"


def train(csv_path: str | None = None, sample_rows: int = 6000) -> dict:
    frame, source = load_training_data(csv_path, sample_rows)
    x, y = prepare_features(frame), frame["trip_duration"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=3, n_jobs=1, random_state=42
    )
    model.fit(x_train, np.log1p(y_train))
    prediction = np.expm1(model.predict(x_test)).clip(1)
    baseline = np.full(len(y_test), y_train.median())
    metrics = {
        "source": source,
        "rows": int(len(frame)),
        "test_rows": int(len(y_test)),
        "mae_seconds": round(float(mean_absolute_error(y_test, prediction)), 1),
        "rmsle": round(float(np.sqrt(mean_squared_log_error(y_test, prediction))), 4),
        "r2": round(float(r2_score(y_test, prediction)), 4),
        "baseline_mae_seconds": round(float(mean_absolute_error(y_test, baseline)), 1),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_duration(payload: dict, bundle: dict) -> tuple[float, float]:
    pickup = pd.Timestamp(payload["pickup_datetime"])
    frame = pd.DataFrame([{
        "pickup_datetime": pickup,
        "pickup_longitude": float(payload["pickup_longitude"]),
        "pickup_latitude": float(payload["pickup_latitude"]),
        "dropoff_longitude": float(payload["dropoff_longitude"]),
        "dropoff_latitude": float(payload["dropoff_latitude"]),
        "passenger_count": int(payload.get("passenger_count", 1)),
    }])
    features = prepare_features(frame)
    seconds = float(np.expm1(bundle["model"].predict(features)[0]))
    return max(60.0, seconds), float(features["distance_km"].iloc[0])
