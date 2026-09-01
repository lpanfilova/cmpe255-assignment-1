from crispdm.pipeline import association_rules, load_data, lsh_search, run_workflows


def test_dataset_contract_and_all_workflows(tmp_path):
    frame = load_data()
    assert len(frame) == 891
    assert frame.PassengerId.is_unique
    result = run_workflows(output_path=tmp_path / "result.json")
    assert result["understanding"]["rows"] == 891
    assert 2 <= result["clustering"]["best_k"] <= 5
    assert 0 < result["anomalies"]["rate"] < 0.1
    assert set(result["supervised"]["models"]) == {"Logistic regression", "Random forest"}
    assert all(0.5 < values["roc_auc"] <= 1 for values in result["supervised"]["models"].values())
    assert result["associations"]["rules"]
    assert result["lsh"]["candidates_examined"] < result["lsh"]["total_rows"]


def test_rule_thresholds_and_lsh_neighbors():
    frame = load_data()
    rules = association_rules(frame, min_support=0.1, min_confidence=0.7)["rules"]
    assert all(rule["support"] >= 0.1 and rule["confidence"] >= 0.7 and rule["lift"] > 1 for rule in rules)
    search = lsh_search(frame, query_index=1)
    assert search["query"]["passenger_id"] == 2
    assert all(0 <= row["similarity"] <= 1 for row in search["neighbors"])

