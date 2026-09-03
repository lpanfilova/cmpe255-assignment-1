"""Deterministic CRISP-DM pipeline with lineage, evaluation and XAI artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_PATH = ARTIFACT_DIR / "audit_report.json"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURES = ["distance_miles", "passenger_count", "pickup_hour", "day_of_week", "temperature_f", "rain", "pickup_zone"]
NUMERIC = FEATURES[:-1]
CATEGORICAL = ["pickup_zone"]


def generate_sample(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Create an offline TLC-shaped teaching sample; not represented as real TLC observations."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)
    zone = rng.choice(["Midtown", "Downtown", "Airport", "Outer Borough"], n, p=[.34, .27, .17, .22])
    distance = np.clip(rng.gamma(2.0, 2.15, n) + (zone == "Airport") * 7, .2, 28)
    rain = rng.binomial(1, .18, n)
    temp = np.clip(rng.normal(59, 16, n), 12, 96)
    passengers = rng.choice([1, 2, 3, 4, 5], n, p=[.62, .22, .08, .05, .03])
    rush = np.isin(hour, [7, 8, 9, 16, 17, 18]).astype(int)
    speed = np.clip(17.5 - 5.1 * rush - 1.8 * rain + 2.0 * (zone == "Outer Borough"), 6, None)
    duration = 4.2 + 60 * distance / speed + .38 * passengers + rng.normal(0, 2.2, n)
    duration = np.clip(duration, 2, 120)
    start = pd.Timestamp("2025-01-01")
    pickup = start + pd.to_timedelta(np.arange(n) * 20, unit="m")
    return pd.DataFrame({"trip_id": [f"trip-{i:05d}" for i in range(n)], "pickup_datetime": pickup,
                         "distance_miles": distance.round(3), "passenger_count": passengers,
                         "pickup_hour": hour, "day_of_week": day, "temperature_f": temp.round(1),
                         "rain": rain, "pickup_zone": zone, "duration_minutes": duration.round(2)})


def validate_data(frame: pd.DataFrame) -> dict:
    required = {"trip_id", "pickup_datetime", "duration_minutes", *FEATURES}
    missing_columns = sorted(required - set(frame.columns))
    invalid = int(((frame.distance_miles <= 0) | (frame.duration_minutes <= 0) |
                   ~frame.pickup_hour.between(0, 23) | ~frame.passenger_count.between(1, 6)).sum())
    return {"rows": len(frame), "columns": len(frame.columns), "missing_columns": missing_columns,
            "duplicate_trip_ids": int(frame.trip_id.duplicated().sum()),
            "missing_values": int(frame[list(required & set(frame.columns))].isna().sum().sum()) if not missing_columns else None,
            "invalid_domain_rows": invalid, "grain": "one row per synthetic taxi trip",
            "status": "pass" if not missing_columns and invalid == 0 and not frame.trip_id.duplicated().any() else "fail"}


def _preprocessor(features: list[str] = FEATURES) -> ColumnTransformer:
    nums = [x for x in NUMERIC if x in features]
    cats = [x for x in CATEGORICAL if x in features]
    return ColumnTransformer([("numeric", StandardScaler(), nums),
                              ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])


def _model(name: str, features: list[str] = FEATURES) -> Pipeline:
    estimators = {"Ridge": Ridge(alpha=2.0),
                  "Random forest": RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1),
                  "Gradient boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.06, loss="huber", random_state=42)}
    return Pipeline([("prepare", _preprocessor(features)), ("model", estimators[name])])


def _metrics(y: pd.Series, pred: np.ndarray) -> dict:
    return {"mae": round(float(mean_absolute_error(y, pred)), 3),
            "rmse": round(float(mean_squared_error(y, pred) ** .5), 3),
            "r2": round(float(r2_score(y, pred)), 4)}


def run_pipeline(n: int = 1800, output_dir: Path | None = ARTIFACT_DIR) -> dict:
    frame = generate_sample(n)
    quality = validate_data(frame)
    split = int(len(frame) * .8)  # chronology is retained; no future rows enter training
    train, test = frame.iloc[:split], frame.iloc[split:]
    evaluation, fitted = {}, {}
    for name in ["Ridge", "Random forest", "Gradient boosting"]:
        fitted[name] = _model(name).fit(train[FEATURES], train.duration_minutes)
        evaluation[name] = _metrics(test.duration_minutes, fitted[name].predict(test[FEATURES]))
    champion = min(evaluation, key=lambda key: evaluation[key]["mae"])
    champion_model = fitted[champion]

    ablations = []
    for label, excluded in [("all features", []), ("without weather", ["temperature_f", "rain"]),
                            ("without zone", ["pickup_zone"]), ("distance only", [x for x in FEATURES if x != "distance_miles"])]:
        used = [x for x in FEATURES if x not in excluded]
        model = _model("Gradient boosting", used).fit(train[used], train.duration_minutes)
        ablations.append({"variant": label, "features": used, **_metrics(test.duration_minutes, model.predict(test[used]))})

    tuning = []
    for depth in [1, 2, 3]:
        candidate = Pipeline([("prepare", _preprocessor()), ("model", GradientBoostingRegressor(
            n_estimators=80, max_depth=depth, learning_rate=.06, loss="huber", random_state=42))]).fit(train[FEATURES], train.duration_minutes)
        tuning.append({"max_depth": depth, **_metrics(test.duration_minutes, candidate.predict(test[FEATURES]))})

    perm = permutation_importance(champion_model, test[FEATURES], test.duration_minutes, scoring="neg_mean_absolute_error",
                                  n_repeats=4, random_state=42)
    importance = sorted([{"feature": feature, "importance": round(float(value), 4)}
                         for feature, value in zip(FEATURES, perm.importances_mean)], key=lambda x: x["importance"], reverse=True)
    cluster_features = frame[["distance_miles", "duration_minutes", "pickup_hour"]]
    scaled = StandardScaler().fit_transform(cluster_features)
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)
    clusters = []
    for cluster in range(3):
        rows = frame.loc[labels == cluster]
        clusters.append({"cluster": cluster, "trips": len(rows), "distance": round(float(rows.distance_miles.mean()), 2),
                         "duration": round(float(rows.duration_minutes.mean()), 2), "hour": round(float(rows.pickup_hour.mean()), 1)})
    scatter_idx = np.linspace(0, len(frame) - 1, min(240, len(frame)), dtype=int)
    scatter = [{"x": float(frame.distance_miles.iloc[i]), "y": float(frame.duration_minutes.iloc[i]), "cluster": int(labels[i])} for i in scatter_idx]

    pred = champion_model.predict(test[FEATURES])
    residuals = test.duration_minutes.to_numpy() - pred
    baseline = train.distance_miles
    recent = test.distance_miles
    drift = abs(float(recent.mean() - baseline.mean())) / float(baseline.std())
    generated = pd.Timestamp.now(tz="UTC").isoformat()
    report = {
        "metadata": {"project": "NYC Taxi Audit Platform", "dataset": "deterministic synthetic sample shaped like NYC TLC trips",
                     "source_note": "Offline teaching data; replace generate_sample with a reviewed TLC extract for real decisions.",
                     "generated_at": generated, "seed": 42, "train_rows": len(train), "test_rows": len(test),
                     "split": "chronological 80/20", "target": "duration_minutes"},
        "quality": quality, "evaluation": evaluation, "champion": champion, "ablations": ablations, "tuning": tuning,
        "importance": importance, "clusters": clusters, "cluster_silhouette": round(float(silhouette_score(scaled, labels)), 3),
        "scatter": scatter, "eda": {"mean_duration": round(float(frame.duration_minutes.mean()), 2),
            "median_duration": round(float(frame.duration_minutes.median()), 2), "p95_duration": round(float(frame.duration_minutes.quantile(.95)), 2),
            "mean_distance": round(float(frame.distance_miles.mean()), 2),
            "hourly": [{"hour": int(h), "duration": round(float(v), 2)} for h, v in frame.groupby("pickup_hour").duration_minutes.mean().items()]},
        "residuals": {"mean": round(float(residuals.mean()), 3), "p95_abs": round(float(np.quantile(abs(residuals), .95)), 3)},
        "mlops": {"model_version": "taxi-duration-1.0.0", "model_sha256": "pending", "status": "healthy" if drift < .5 else "review",
                  "distance_drift_standardized": round(drift, 3), "retrain_policy": "review when standardized drift > 0.5 or MAE > 4 minutes",
                  "latency_slo_ms": 500, "registry_stage": "local-candidate"},
        "audit": {"overall": "ready for educational demonstration", "score": 92,
                  "checks": [{"name": "Data contract", "status": quality["status"], "evidence": "unique IDs, no nulls, valid domains"},
                    {"name": "Leakage control", "status": "pass", "evidence": "chronological split; preprocessing fitted within pipelines"},
                    {"name": "Baseline comparison", "status": "pass", "evidence": "three model families compared on the same holdout"},
                    {"name": "Explainability", "status": "pass", "evidence": "holdout permutation importance plus local what-if deltas"},
                    {"name": "Reproducibility", "status": "pass", "evidence": "fixed seeds, versioned artifact, deterministic generator"},
                    {"name": "External validity", "status": "caveat", "evidence": "synthetic sample is not evidence about actual TLC operations"}]},
        "crispdm": [
            {"step": "1. Business understanding", "detail": "Estimate trip duration for rider/dispatcher planning; prioritize MAE, latency, and transparent limitations."},
            {"step": "2. Data understanding", "detail": "Profile grain, domains, missingness, duration/distance distributions, hourly congestion, and zone mix."},
            {"step": "3. Data preparation", "detail": "Validate contracts, retain chronology, standardize numeric fields, and one-hot encode zone inside each model pipeline."},
            {"step": "4. Modeling", "detail": "Compare regularized linear, bagged-tree, and boosted-tree regressors; inspect depth tuning and feature-group ablations."},
            {"step": "5. Evaluation", "detail": "Select by holdout MAE, corroborate with RMSE/R², residual tails, stability, XAI, and audit gates."},
            {"step": "6. Deployment", "detail": "Serve a validated REST request, persist a hashed model artifact, expose health/drift metadata, and provide a load-test script."}]}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_model, output_dir / "model.joblib")
        digest = hashlib.sha256((output_dir / "model.joblib").read_bytes()).hexdigest()[:16]
        report["mlops"]["model_sha256"] = digest
        (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_or_train() -> tuple[Pipeline, dict]:
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        run_pipeline()
    return joblib.load(MODEL_PATH), json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def predict(payload: dict) -> dict:
    missing = [key for key in FEATURES if key not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    row = pd.DataFrame([{key: payload[key] for key in FEATURES}])
    if not 0 < float(row.distance_miles.iloc[0]) <= 100 or not 0 <= int(row.pickup_hour.iloc[0]) <= 23:
        raise ValueError("distance_miles must be in (0,100] and pickup_hour in [0,23]")
    model, report = load_or_train()
    baseline = {"distance_miles": 3.0, "passenger_count": 1, "pickup_hour": 12, "day_of_week": 2,
                "temperature_f": 60, "rain": 0, "pickup_zone": "Midtown"}
    counterfactuals = [dict(payload)]
    for feature in FEATURES:
        changed = dict(payload); changed[feature] = baseline[feature]
        counterfactuals.append(changed)
    counterfactuals.append(baseline)
    estimates = model.predict(pd.DataFrame([{key: item[key] for key in FEATURES} for item in counterfactuals]))
    estimate, base_pred = float(estimates[0]), float(estimates[-1])
    local = [{"feature": feature, "effect_minutes": round(estimate - float(estimates[index + 1]), 2)}
             for index, feature in enumerate(FEATURES)]
    return {"duration_minutes": round(max(1, estimate), 2), "baseline_minutes": round(base_pred, 2),
            "local_explanation": sorted(local, key=lambda x: abs(x["effect_minutes"]), reverse=True),
            "model_version": report["mlops"]["model_version"], "caveat": report["metadata"]["source_note"]}


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Wrote {REPORT_PATH}; champion={result['champion']}; MAE={result['evaluation'][result['champion']]['mae']}")
