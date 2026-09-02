import numpy as np

from forecasting.pipeline import HOLDOUT, load_data, run_pipeline


def test_data_contract_and_pipeline(tmp_path):
    series = load_data()
    assert len(series) == 144
    assert series.index.is_monotonic_increasing and not series.isna().any()
    result = run_pipeline(output_path=tmp_path / "forecast.json")
    assert result["metadata"]["holdout_rows"] == HOLDOUT
    assert set(result["evaluation"]) == {"Seasonal naive", "Holt-Winters", "Ridge trend + seasonality"}
    assert result["champion"] in result["evaluation"]
    assert result["evaluation"][result["champion"]]["mase"] < 1
    assert len(result["forecast"]["values"]) == 12
    assert np.all(np.asarray(result["forecast"]["lower"]) >= 0)


def test_champion_is_selected_by_mase(tmp_path):
    result = run_pipeline(output_path=tmp_path / "forecast.json")
    expected = min(result["evaluation"], key=lambda name: result["evaluation"][name]["mase"])
    assert result["champion"] == expected

