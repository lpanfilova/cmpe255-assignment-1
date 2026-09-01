import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app
from model import load_data, train


def test_data_contract():
    frame = load_data()
    assert len(frame) == 200
    assert frame["CustomerID"].is_unique


def test_pipeline_and_dashboard(tmp_path):
    metrics = train(artifact_dir=tmp_path, stability_rounds=1)
    assert 2 <= metrics["winner"]["clusters"] <= 9
    assert metrics["winner"]["silhouette"] > 0.3
    assert len(metrics["experiments"]) == 48
    assert (tmp_path / "model.joblib").exists()
    client = create_app(tmp_path).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json["artifacts_ready"] is True
    response = client.get("/api/dashboard?gender=Female")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filtered_rows"] > 0
    assert all(point["Gender"] == "Female" for point in payload["points"])
