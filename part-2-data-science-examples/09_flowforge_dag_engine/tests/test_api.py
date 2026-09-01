from app import create_app


SAMPLE = {
    "name": "API workflow",
    "nodes": [
        {"id": "source", "type": "source", "config": {"value": [{"value": 3}]}},
        {"id": "sum", "type": "aggregate", "config": {"field": "value", "operation": "sum"}},
    ],
    "edges": [{"source": "source", "target": "sum"}],
}


def test_full_http_workflow_lifecycle(tmp_path):
    client = create_app(tmp_path / "test.db").test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["status"] == "ok"

    validation = client.post("/api/validate", json=SAMPLE)
    assert validation.status_code == 200
    assert validation.get_json()["plan"] == ["source", "sum"]

    saved = client.post("/api/workflows", json=SAMPLE)
    assert saved.status_code == 201
    definition = saved.get_json()
    assert definition["version"] == 1
    assert client.get(f"/api/workflows/{definition['id']}").get_json()["name"] == "API workflow"

    completed = client.post(f"/api/workflows/{definition['id']}/runs")
    assert completed.status_code == 201
    assert completed.get_json()["outcomes"][-1]["output"]["value"] == 3
    assert len(client.get(f"/api/workflows/{definition['id']}/runs").get_json()) == 1


def test_invalid_workflow_returns_actionable_client_error(tmp_path):
    client = create_app(tmp_path / "test.db").test_client()
    response = client.post("/api/validate", json={"nodes": [], "edges": []})
    assert response.status_code == 400
    assert "at least one node" in response.get_json()["error"]

