from app import app


def test_dashboard_api_and_health():
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"CRISP-DM playbook" in page.data and b"Admin & governance" in page.data
    payload = client.get("/api/forecast").get_json()
    assert payload["metadata"]["rows"] == 144
    health = client.get("/health").get_json()
    assert health["status"] == "healthy" and health["artifact"] is True
