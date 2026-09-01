"""Reproducible Annthyroid anomaly detection and bounded autoresearch."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "annthyroid.mat"
FEATURES = ["TSH", "T3", "TT4", "T4U", "FTI", "age"]


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    parameter: float


def load_data(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    raw = loadmat(path)
    if "X" not in raw or "y" not in raw:
        raise ValueError("MAT file must contain X and y arrays.")
    x, y = np.asarray(raw["X"], float), np.asarray(raw["y"]).reshape(-1).astype(int)
    if x.ndim != 2 or x.shape[1] != 6 or len(x) != len(y):
        raise ValueError("Expected aligned X[n,6] and y[n] arrays.")
    if not np.isfinite(x).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Features must be finite and labels binary.")
    frame = pd.DataFrame(x, columns=FEATURES)
    frame.insert(0, "record_id", np.arange(1, len(frame) + 1))
    frame["is_anomaly"] = y
    return frame


def split_indices(y: np.ndarray, seed: int = 42):
    """Deterministic stratified 60/20/20 split without requiring labels at scoring time."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        a, b = int(.6 * len(idx)), int(.8 * len(idx))
        for target, values in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            target.extend(values.tolist())
    return tuple(np.array(sorted(part)) for part in parts)


def candidate_space():
    return [Candidate("robust_z", q) for q in (0.95, 0.975)] + \
        [Candidate("isolation_forest", n) for n in (64, 128, 256, 384)] + \
        [Candidate("lof", n) for n in (10, 20, 35, 50)] + \
        [Candidate("one_class_svm", g) for g in (0.01, 0.03, 0.1, 0.3)]


def fit_detector(candidate: Candidate, x_normal: np.ndarray, seed: int = 42):
    if candidate.algorithm == "robust_z":
        scaler = RobustScaler().fit(x_normal)
        return {"scaler": scaler, "quantile": candidate.parameter}
    if candidate.algorithm == "isolation_forest":
        return Pipeline([("scale", RobustScaler()), ("model", IsolationForest(
            n_estimators=int(candidate.parameter), max_samples="auto", contamination="auto",
            random_state=seed, n_jobs=1))]).fit(x_normal)
    if candidate.algorithm == "lof":
        return Pipeline([("scale", RobustScaler()), ("model", LocalOutlierFactor(
            n_neighbors=int(candidate.parameter), novelty=True, contamination="auto"))]).fit(x_normal)
    return Pipeline([("scale", StandardScaler()), ("model", OneClassSVM(
        kernel="rbf", gamma=candidate.parameter, nu=0.03, cache_size=256))]).fit(x_normal)


def anomaly_score(model, candidate: Candidate, x: np.ndarray) -> np.ndarray:
    if candidate.algorithm == "robust_z":
        z = np.abs(model["scaler"].transform(x))
        return np.quantile(z, model["quantile"], axis=1)
    return -np.asarray(model.decision_function(x)).reshape(-1)


def _metrics(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "alert_rate": float(pred.mean()), "true_positive": int(tp), "false_positive": int(fp),
            "false_negative": int(fn), "true_negative": int(tn)}


def search(frame: pd.DataFrame, seed: int = 42, alert_budget: float = .05):
    x, y = frame[FEATURES].to_numpy(), frame["is_anomaly"].to_numpy()
    train_idx, valid_idx, test_idx = split_indices(y, seed)
    x_normal = x[train_idx][y[train_idx] == 0]
    cache = {}
    for candidate in candidate_space():
        started = time.perf_counter()
        model = fit_detector(candidate, x_normal, seed)
        valid_scores = anomaly_score(model, candidate, x[valid_idx])
        latency = (time.perf_counter() - started) * 1000
        threshold = float(np.quantile(valid_scores, 1 - alert_budget))
        row = {**asdict(candidate), **_metrics(y[valid_idx], valid_scores, threshold),
               "threshold": threshold, "fit_score_ms": latency}
        if candidate.algorithm == "isolation_forest":
            alternate = fit_detector(candidate, x_normal, seed + 1)
            stability = spearmanr(valid_scores, anomaly_score(alternate, candidate, x[valid_idx])).statistic
        else:
            stability = 1.0
        row["stability"] = float(np.nan_to_num(stability))
        row["objective"] = .45 * row["pr_auc"] + .25 * row["roc_auc"] + .20 * row["recall"] + .10 * row["stability"] - .02 * min(latency / 1000, 1)
        cache[candidate] = row

    # Greedy hill climb over adjacent parameters plus same-position algorithm swaps.
    space = candidate_space()
    current, path = Candidate("isolation_forest", 128), []
    while True:
        index = space.index(current)
        neighbors = {current, space[max(0, index - 1)], space[min(len(space) - 1, index + 1)]}
        family_pos = [c for c in space if c.algorithm == current.algorithm].index(current)
        for family in {c.algorithm for c in space}:
            options = [c for c in space if c.algorithm == family]
            neighbors.add(options[min(family_pos, len(options) - 1)])
        nxt = max(neighbors, key=lambda c: cache[c]["objective"])
        path.append({**cache[nxt], "iteration": len(path)})
        if nxt == current:
            break
        current = nxt
    winner = max(cache, key=lambda c: cache[c]["objective"])
    return winner, sorted(cache.values(), key=lambda r: r["objective"], reverse=True), path, (train_idx, valid_idx, test_idx)


def train(data_path=DEFAULT_DATA, artifact_dir=ROOT / "artifacts", seed=42, alert_budget=.05):
    frame = load_data(data_path)
    winner, leaderboard, path, splits = search(frame, seed, alert_budget)
    train_idx, valid_idx, test_idx = splits
    x, y = frame[FEATURES].to_numpy(), frame["is_anomaly"].to_numpy()
    normal_fit = train_idx[y[train_idx] == 0]
    model = fit_detector(winner, x[normal_fit], seed)
    validation_scores = anomaly_score(model, winner, x[valid_idx])
    threshold = float(np.quantile(validation_scores, 1 - alert_budget))
    test_scores = anomaly_score(model, winner, x[test_idx])
    test_metrics = _metrics(y[test_idx], test_scores, threshold)
    precision, recall, _ = precision_recall_curve(y[test_idx], test_scores)
    scored = frame.iloc[test_idx].copy()
    scored["anomaly_score"] = test_scores
    scored["flagged"] = test_scores >= threshold
    scored = scored.sort_values("anomaly_score", ascending=False)
    artifact_dir = Path(artifact_dir); artifact_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(artifact_dir / "scored_holdout.csv", index=False)
    joblib.dump({"candidate": winner, "detector": model, "threshold": threshold, "features": FEATURES}, artifact_dir / "model.joblib")
    clean = lambda obj: json.loads(json.dumps(obj, default=lambda v: float(v) if isinstance(v, np.generic) else v))
    metrics = {"dataset": {"name": "Annthyroid anomaly benchmark", "rows": len(frame), "features": FEATURES,
                "anomalies": int(y.sum()), "prevalence": float(y.mean()), "missing": int(frame.isna().sum().sum()),
                "split": {"train": len(train_idx), "validation": len(valid_idx), "holdout": len(test_idx)},
                "source": "Kaggle/UCI Annthyroid; ODDS six-numeric-feature benchmark"},
               "winner": {**asdict(winner), "threshold": threshold, **test_metrics},
               "research": {"objective": "0.45 PR-AUC + 0.25 ROC-AUC + 0.20 recall@5% + 0.10 rank stability - latency penalty",
                "alert_budget": alert_budget, "evaluated_configurations": len(leaderboard), "path": path,
                "global_optimum_reached": winner == Candidate(path[-1]["algorithm"], path[-1]["parameter"])},
               "experiments": leaderboard,
               "score_distribution": {"normal_median": float(np.median(test_scores[y[test_idx] == 0])),
                "anomaly_median": float(np.median(test_scores[y[test_idx] == 1])),
                "min": float(test_scores.min()), "max": float(test_scores.max())},
               "pr_curve": [{"recall": float(r), "precision": float(p)} for p, r in zip(precision[::max(1, len(precision)//50)], recall[::max(1, len(recall)//50)])],
               "methodology": {"seed": seed, "fit_population": "normal-only training split",
                "threshold_policy": "95th percentile validation score", "holdout_used_for_search": False}}
    (artifact_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2), encoding="utf-8")
    return clean(metrics)
