from taxi_audit.pipeline import FEATURES, generate_sample, predict, run_pipeline, validate_data


def test_data_contract_and_full_pipeline(tmp_path):
    frame = generate_sample(600)
    assert validate_data(frame)["status"] == "pass"
    result = run_pipeline(n=600, output_dir=tmp_path)
    assert set(result["evaluation"]) == {"Ridge", "Random forest", "Gradient boosting"}
    assert result["champion"] in result["evaluation"]
    assert result["evaluation"][result["champion"]]["mae"] < 5
    assert len(result["clusters"]) == 3 and len(result["ablations"]) == 4
    assert (tmp_path / "model.joblib").exists() and result["mlops"]["model_sha256"] != "pending"


def test_inference_contract_and_explanation():
    payload = dict(zip(FEATURES, [4.2, 2, 17, 4, 71, 0, "Midtown"]))
    result = predict(payload)
    assert result["duration_minutes"] > 0
    assert len(result["local_explanation"]) == len(FEATURES)

