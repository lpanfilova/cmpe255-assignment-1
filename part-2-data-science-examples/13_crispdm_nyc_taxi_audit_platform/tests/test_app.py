from app import app


PAYLOAD = {"distance_miles": 4.2, "passenger_count": 2, "pickup_hour": 17, "day_of_week": 4,
           "temperature_f": 71, "rain": 0, "pickup_zone": "Midtown"}


def test_dashboard_health_audit_and_prediction():
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"Evidence, not just predictions" in page.data and b"CRISP-DM decision trail" in page.data
    assert client.get("/health").get_json()["status"] in {"healthy", "review"}
    assert client.get("/api/v1/audit").get_json()["audit"]["score"] >= 90
    response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 200 and response.get_json()["duration_minutes"] > 0


def test_bad_inference_is_rejected():
    response = app.test_client().post("/api/v1/predict", json={"distance_miles": -1})
    assert response.status_code == 400 and "Missing fields" in response.get_json()["error"]

