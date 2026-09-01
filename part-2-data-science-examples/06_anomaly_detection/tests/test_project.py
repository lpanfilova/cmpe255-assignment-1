import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from model import FEATURES, load_data, split_indices, train

def test_data_contract_and_stratified_split():
    frame = load_data()
    assert frame.shape == (7200, 8) and frame[FEATURES].isna().sum().sum() == 0
    parts = split_indices(frame.is_anomaly.to_numpy())
    assert len(set(np.concatenate(parts))) == len(frame)
    assert all(frame.iloc[p].is_anomaly.sum() > 0 for p in parts)

def test_pipeline_and_dashboard(tmp_path):
    metrics = train(artifact_dir=tmp_path)
    assert metrics["winner"]["pr_auc"] > metrics["dataset"]["prevalence"]
    assert metrics["winner"]["roc_auc"] > .7
    assert metrics["methodology"]["holdout_used_for_search"] is False
    client = create_app(tmp_path).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["artifacts_ready"] is True
    payload = client.get("/api/dashboard?flagged=true&limit=10").get_json()
    assert payload["records"] and all(r["flagged"] for r in payload["records"])
    assert len(payload["records"]) <= 10
