import json

from app import app
from lab import discover_skills, run


def test_all_vendored_skills_are_discovered():
    skills = discover_skills()
    assert len(skills) == 46
    assert {s.collection for s in skills} == {"param087", "nimrodfisher"}
    assert len({(s.collection, s.name) for s in skills}) == 46


def test_lab_generates_every_demonstration(tmp_path):
    manifest = run(tmp_path)
    reports = list((tmp_path / "skills").glob("*.md"))
    assert len(reports) == manifest["skill_count"] == 46
    assert manifest["dataset"]["rows"] == 891
    assert manifest["model_metrics"]["roc_auc"] > 0.75
    assert (tmp_path / "titanic_pipeline.joblib").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["skill_count"] == 46


def test_application_health_and_prediction():
    client = app.test_client()
    assert client.get("/api/health").get_json() == {"status": "ok", "skills": 46}
    response = client.post("/api/predict", json={"Pclass": 1, "Sex": "female", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "S"})
    assert response.status_code == 200
    assert 0 <= response.get_json()["survival_probability"] <= 1


def test_prediction_validation():
    response = app.test_client().post("/api/predict", json={"Pclass": 1})
    assert response.status_code == 400
    assert "Sex" in response.get_json()["fields"]
