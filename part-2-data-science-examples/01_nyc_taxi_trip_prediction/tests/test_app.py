import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app


def test_health_and_home():
    client = create_app(testing=True).test_client()
    assert client.get("/health").get_json() == {"model_loaded": True, "status": "ok"}
    assert b"Know the ride" in client.get("/").data


def test_prediction_is_plausible():
    client = create_app(testing=True).test_client()
    response = client.post("/predict", json={
        "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7128, "dropoff_longitude": -74.0060,
        "pickup_datetime": "2026-09-01T17:30", "passenger_count": 2,
    })
    data = response.get_json()
    assert response.status_code == 200
    assert 3 < data["duration_minutes"] < 90
    assert data["estimated_route_km"] > data["straight_line_km"] > 0


def test_rejects_outside_nyc():
    client = create_app(testing=True).test_client()
    response = client.post("/predict", json={
        "pickup_latitude": 34.0, "pickup_longitude": -118.2,
        "dropoff_latitude": 40.7, "dropoff_longitude": -74.0,
        "pickup_datetime": "2026-09-01T12:00",
    })
    assert response.status_code == 400

