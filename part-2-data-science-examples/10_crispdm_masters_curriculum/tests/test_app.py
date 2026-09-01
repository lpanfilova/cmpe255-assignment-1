from app import app


def test_dashboard_api_and_health():
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"CRISP-DM" in page.data and b"MinHash LSH" in page.data
    payload = client.get("/api/results").get_json()
    assert payload["understanding"]["rows"] == 891
    assert client.get("/health").get_json()["status"] == "ok"

