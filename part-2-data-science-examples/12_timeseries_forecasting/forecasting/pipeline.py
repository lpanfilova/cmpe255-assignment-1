"""Leakage-safe, laptop-sized forecasting pipeline for monthly airline demand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "air_passengers.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "forecast.json"
SEASON_LENGTH = 12
HOLDOUT = 24
HORIZON = 12


def load_data(path: Path = DATA_PATH) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["month"])
    if set(frame.columns) != {"month", "passengers"}:
        raise ValueError("Dataset must contain month and passengers columns")
    if frame.month.duplicated().any() or frame.passengers.isna().any() or (frame.passengers <= 0).any():
        raise ValueError("Months must be unique and passenger counts must be positive")
    frame = frame.sort_values("month")
    expected = pd.date_range(frame.month.min(), frame.month.max(), freq="MS")
    if len(frame) != len(expected) or not frame.month.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("Monthly series must be complete and contiguous")
    series = frame.set_index("month")["passengers"].astype(float)
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series


def _features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    t = ((index.year - origin.year) * 12 + index.month - origin.month).to_numpy()
    month = index.month.to_numpy()
    return np.column_stack([t, t**2, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                            *[(month == value).astype(int) for value in range(2, 13)]])


def _ridge_forecast(train: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    model = Ridge(alpha=1.0).fit(_features(train.index, train.index[0]), np.log(train.to_numpy()))
    return np.exp(model.predict(_features(index, train.index[0])))


def _forecast(model: str, train: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    if model == "Seasonal naive":
        return np.resize(train.iloc[-SEASON_LENGTH:].to_numpy(), len(index))
    if model == "Holt-Winters":
        fitted = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=SEASON_LENGTH,
                                      initialization_method="estimated").fit(optimized=True)
        return fitted.forecast(len(index)).to_numpy()
    if model == "Ridge trend + seasonality":
        return _ridge_forecast(train, index)
    raise ValueError(f"Unknown model: {model}")


def _metrics(actual: np.ndarray, predicted: np.ndarray, naive_scale: float) -> dict:
    errors = actual - predicted
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "mape": round(float(np.mean(np.abs(errors / actual)) * 100), 2),
        "mase": round(float(np.mean(np.abs(errors)) / naive_scale), 3),
        "bias": round(float(np.mean(predicted - actual)), 2),
    }


def run_pipeline(data_path: Path = DATA_PATH, output_path: Path | None = ARTIFACT_PATH) -> dict:
    series = load_data(data_path)
    train, test = series.iloc[:-HOLDOUT], series.iloc[-HOLDOUT:]
    naive_scale = float(np.mean(np.abs(train.to_numpy()[SEASON_LENGTH:] - train.to_numpy()[:-SEASON_LENGTH])))
    models = ["Seasonal naive", "Holt-Winters", "Ridge trend + seasonality"]
    evaluation, predictions = {}, {}
    for name in models:
        values = _forecast(name, train, test.index)
        evaluation[name] = _metrics(test.to_numpy(), values, naive_scale)
        predictions[name] = [round(float(value), 1) for value in values]
    champion = min(models, key=lambda name: evaluation[name]["mase"])
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
    future = _forecast(champion, series, future_index)
    residuals = test.to_numpy() - np.asarray(predictions[champion])
    interval_width = 1.96 * float(np.std(residuals, ddof=1))
    result = {
        "metadata": {"dataset": "International Airline Passengers", "source": "Box & Jenkins / R datasets AirPassengers",
                     "frequency": "monthly", "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                     "rows": len(series), "train_rows": len(train), "holdout_rows": len(test)},
        "quality": {"missing_months": 0, "missing_values": 0, "duplicate_months": 0,
                    "start": series.index[0].strftime("%Y-%m"), "end": series.index[-1].strftime("%Y-%m")},
        "series": {"dates": [date.strftime("%Y-%m") for date in series.index],
                   "actual": [int(value) for value in series], "test_start": test.index[0].strftime("%Y-%m")},
        "evaluation": evaluation, "holdout_predictions": predictions, "champion": champion,
        "forecast": {"dates": [date.strftime("%Y-%m") for date in future_index],
                     "values": [round(float(value), 1) for value in future],
                     "lower": [round(max(0.0, float(value - interval_width)), 1) for value in future],
                     "upper": [round(float(value + interval_width), 1) for value in future],
                     "interval": "Approximate 95% residual interval"},
        "seasonality": {str(month): round(float(series[series.index.month == month].mean()), 1) for month in range(1, 13)},
        "operations": {"status": "healthy", "drift_status": "baseline", "refresh": "manual local run",
                       "policy": "Retrain after each complete month; review if MASE > 1 or bias exceeds 10% of mean demand."},
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Wrote {ARTIFACT_PATH}; champion={result['champion']}; MASE={result['evaluation'][result['champion']]['mase']}")
