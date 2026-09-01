"""Reproducible customer-segmentation training and autoresearch utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Avoid MKL's small-dataset Windows thread overhead and known K-Means leak.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "Mall_Customers.csv"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    clusters: int
    scaler: str


def load_data(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"Genre": "Gender"})
    required = {"CustomerID", "Gender", *FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Dataset must be non-empty and contain no missing required values.")
    if frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID values must be unique.")
    if not frame["Gender"].isin(["Male", "Female"]).all():
        raise ValueError("Gender values must be Male or Female.")
    if (frame[FEATURES] < 0).any().any():
        raise ValueError("Numeric customer features must be non-negative.")
    return frame


def _cluster(candidate: Candidate, values: np.ndarray, seed: int) -> np.ndarray:
    if candidate.algorithm == "kmeans":
        return KMeans(candidate.clusters, n_init=10, random_state=seed).fit_predict(values)
    if candidate.algorithm == "gmm":
        return GaussianMixture(candidate.clusters, covariance_type="full", n_init=2, random_state=seed).fit_predict(values)
    return AgglomerativeClustering(candidate.clusters, linkage="ward").fit_predict(values)


def _stability(candidate: Candidate, values: np.ndarray, labels: np.ndarray, seed: int, rounds: int) -> float:
    """Bootstrap stability: ARI on points included in each resample."""
    rng = np.random.default_rng(seed)
    scores = []
    for offset in range(rounds):
        sample = rng.choice(len(values), len(values), replace=True)
        unique = np.unique(sample)
        sampled_labels = _cluster(candidate, values[sample], seed + offset + 1)
        # A repeated row receives one deterministic label; compare only unique rows.
        first = {idx: sampled_labels[np.flatnonzero(sample == idx)[0]] for idx in unique}
        scores.append(adjusted_rand_score(labels[unique], [first[idx] for idx in unique]))
    return float(np.mean(scores))


def search(frame: pd.DataFrame, seed: int = 42, stability_rounds: int = 3):
    """Hill-climb from k=2, evaluating neighboring k and model/preprocessing moves."""
    raw = frame[FEATURES].to_numpy(float)
    scalers = {"standard": StandardScaler(), "robust": RobustScaler()}
    cache: dict[Candidate, dict] = {}

    def evaluate(c: Candidate):
        if c in cache:
            return cache[c]
        scaled = scalers[c.scaler].fit_transform(raw)
        labels = _cluster(c, scaled, seed)
        sil = float(silhouette_score(scaled, labels))
        db = float(davies_bouldin_score(scaled, labels))
        ch = float(calinski_harabasz_score(scaled, labels))
        stability = _stability(c, scaled, labels, seed, stability_rounds)
        cache[c] = {**asdict(c), "silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "stability_ari": stability,
                    "labels": labels, "scaled": scaled}
        return cache[c]

    current = Candidate("kmeans", 2, "standard")
    path = []
    while True:
        neighborhood = {current}
        for k in (current.clusters - 1, current.clusters + 1):
            if 2 <= k <= 9:
                neighborhood.add(Candidate(current.algorithm, k, current.scaler))
        for algorithm in ("kmeans", "gmm", "agglomerative"):
            neighborhood.add(Candidate(algorithm, current.clusters, current.scaler))
        for scaler in scalers:
            neighborhood.add(Candidate(current.algorithm, current.clusters, scaler))
        rows = [evaluate(c) for c in neighborhood]
        # Research-guided objective: quality + resampling stability; DB is minimized.
        for row in rows:
            row["objective"] = 0.55 * row["silhouette"] + 0.35 * row["stability_ari"] - 0.10 * min(row["davies_bouldin"], 2) / 2
        best = max(rows, key=lambda row: row["objective"])
        path.append({k: v for k, v in best.items() if k not in {"labels", "scaled"}})
        nxt = Candidate(best["algorithm"], best["clusters"], best["scaler"])
        if nxt == current:
            break
        current = nxt

    # Audit the full compact design space as a guard against local maxima.
    for algorithm in ("kmeans", "gmm", "agglomerative"):
        for scaler in scalers:
            for k in range(2, 10):
                row = evaluate(Candidate(algorithm, k, scaler))
                row["objective"] = 0.55 * row["silhouette"] + 0.35 * row["stability_ari"] - 0.10 * min(row["davies_bouldin"], 2) / 2
    winner = max(cache.values(), key=lambda row: row["objective"])
    leaderboard = sorted(cache.values(), key=lambda row: row["objective"], reverse=True)
    return winner, leaderboard, path


def _segment_name(row: pd.Series) -> str:
    income = row["income"]
    spending = row["spending"]
    if income >= 70 and spending >= 60:
        return "Premium champions"
    if income >= 70 and spending < 45:
        return "Affluent cautious"
    if income < 45 and spending >= 60:
        return "Emerging enthusiasts"
    if income < 45 and spending < 45:
        return "Budget minimalists"
    return "Mainstream shoppers"


def train(data_path=DEFAULT_DATA, artifact_dir=ROOT / "artifacts", seed=42, stability_rounds=3):
    frame = load_data(data_path)
    winner, leaderboard, path = search(frame, seed, stability_rounds)
    labels = winner["labels"]
    scaler = StandardScaler() if winner["scaler"] == "standard" else RobustScaler()
    scaled = scaler.fit_transform(frame[FEATURES])
    if winner["algorithm"] == "kmeans":
        estimator = KMeans(winner["clusters"], n_init=10, random_state=seed).fit(scaled)
    elif winner["algorithm"] == "gmm":
        estimator = GaussianMixture(winner["clusters"], n_init=2, random_state=seed).fit(scaled)
    else:
        estimator = AgglomerativeClustering(winner["clusters"], linkage="ward").fit(scaled)

    scored = frame.copy()
    scored["cluster"] = labels.astype(int)
    scored["silhouette"] = silhouette_samples(scaled, labels)
    coords = PCA(n_components=2, random_state=seed).fit_transform(scaled)
    scored[["pca_x", "pca_y"]] = coords
    profiles = scored.groupby("cluster", as_index=False).agg(
        customers=("CustomerID", "size"), age=("Age", "mean"),
        income=("Annual Income (k$)", "mean"), spending=("Spending Score (1-100)", "mean"),
        silhouette=("silhouette", "mean"), female_share=("Gender", lambda s: (s == "Female").mean()),
    )
    profiles["segment"] = profiles.apply(_segment_name, axis=1)
    name_map = profiles.set_index("cluster")["segment"]
    scored["segment"] = scored["cluster"].map(name_map)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(artifact_dir / "customers_segmented.csv", index=False)
    joblib.dump({"scaler": scaler, "estimator": estimator, "features": FEATURES}, artifact_dir / "model.joblib")
    clean_board = [{k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                    for k, v in row.items() if k not in {"labels", "scaled"}} for row in leaderboard]
    metrics = {
        "dataset": {"rows": len(frame), "features": FEATURES, "missing": int(frame.isna().sum().sum()),
                    "duplicates": int(frame.duplicated().sum()), "source": "Kaggle Mall Customers (bundled CSV)"},
        "winner": clean_board[0], "hill_climb_path": path,
        "experiments": clean_board, "profiles": profiles.round(3).to_dict("records"),
        "methodology": {"seed": seed, "stability_rounds": stability_rounds,
                        "objective": "0.55*silhouette + 0.35*bootstrap ARI - 0.10*min(DB,2)/2"},
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
