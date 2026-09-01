"""Budget-aware AutoGluon experiments for three small tabular tasks."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SEED = 42


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    problem_type: str
    metric: str
    label: str = "target"


TASKS = {
    "binary": TaskSpec("binary", "Breast cancer screening", "binary", "roc_auc"),
    "multiclass": TaskSpec("multiclass", "Iris species", "multiclass", "accuracy"),
    "regression": TaskSpec("regression", "Diabetes progression", "regression", "root_mean_squared_error"),
}

# Ordered neighborhood: each accepted candidate becomes the new incumbent.
CANDIDATES = [
    {"name": "fast_baseline", "presets": "medium_quality",
     "hyperparameters": {"RF": {}, "KNN": {}}},
    {"name": "diverse_portfolio", "presets": "medium_quality",
     "hyperparameters": {"RF": {}, "XT": {}, "KNN": {}}},
    {"name": "deployment_optimized", "presets": "medium_quality",
     "hyperparameters": {"RF": {}, "XT": {}}},
]


def load_task(key: str) -> tuple[pd.DataFrame, TaskSpec]:
    """Return a deterministic, local, license-friendly sklearn dataset."""
    spec = TASKS[key]
    loaders = {"binary": load_breast_cancer, "multiclass": load_iris, "regression": load_diabetes}
    bunch = loaders[key](as_frame=True)
    frame = bunch.frame.rename(columns={bunch.frame.columns[-1]: spec.label})
    return frame, spec


def split_data(frame: pd.DataFrame, spec: TaskSpec):
    stratify = frame[spec.label] if spec.problem_type != "regression" else None
    train, test = train_test_split(frame, test_size=.2, random_state=SEED, stratify=stratify)
    train, validation = train_test_split(
        train, test_size=.25, random_state=SEED,
        stratify=train[spec.label] if spec.problem_type != "regression" else None,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def _evaluate(predictor, data: pd.DataFrame, spec: TaskSpec) -> dict[str, float]:
    y = data[spec.label]
    x = data.drop(columns=spec.label)
    started = time.perf_counter()
    pred = predictor.predict(x)
    latency_ms = (time.perf_counter() - started) * 1000 / len(x)
    if spec.problem_type == "binary":
        proba = predictor.predict_proba(x)
        positive = proba.iloc[:, 1] if hasattr(proba, "iloc") else proba[:, 1]
        return {"roc_auc": roc_auc_score(y, positive), "accuracy": accuracy_score(y, pred), "latency_ms_row": latency_ms}
    if spec.problem_type == "multiclass":
        proba = predictor.predict_proba(x)
        return {"accuracy": accuracy_score(y, pred), "log_loss": log_loss(y, proba), "latency_ms_row": latency_ms}
    return {"rmse": mean_squared_error(y, pred) ** .5, "r2": r2_score(y, pred), "latency_ms_row": latency_ms}


def _utility(metrics: dict[str, float], spec: TaskSpec) -> float:
    if spec.problem_type == "binary": return metrics["roc_auc"]
    if spec.problem_type == "multiclass": return metrics["accuracy"]
    return -metrics["rmse"]


def run_task(key: str, artifact_dir: Path, time_limit: int = 15, max_candidates: int = 2) -> dict[str, Any]:
    """Run a bounded greedy search and evaluate only its winner on untouched test data."""
    from autogluon.tabular import TabularPredictor

    frame, spec = load_task(key)
    train, validation, test = split_data(frame, spec)
    task_dir = artifact_dir / "models" / key
    task_dir.mkdir(parents=True, exist_ok=True)
    trials, incumbent, incumbent_utility = [], None, -np.inf

    for index, config in enumerate(CANDIDATES[:max_candidates]):
        path = task_dir / config["name"]
        if path.exists(): shutil.rmtree(path)
        kwargs = {"train_data": train, "tuning_data": validation, "time_limit": time_limit,
                  "presets": config["presets"], "num_cpus": 2, "num_gpus": 0,
                  "verbosity": 0, "fit_weighted_ensemble": True}
        kwargs["hyperparameters"] = config["hyperparameters"]
        started = time.perf_counter()
        predictor = TabularPredictor(
            label=spec.label, problem_type=spec.problem_type, eval_metric=spec.metric,
            path=str(path), verbosity=0,
        ).fit(**kwargs)
        elapsed = time.perf_counter() - started
        validation_metrics = _evaluate(predictor, validation, spec)
        utility = _utility(validation_metrics, spec)
        accepted = utility > incumbent_utility
        if accepted:
            incumbent, incumbent_utility = config["name"], utility
        trials.append({"iteration": index + 1, "candidate": config["name"], "accepted": accepted,
                       "utility": utility, "duration_sec": elapsed, "validation": validation_metrics,
                       "model_count": len(predictor.model_names())})

    winner_path = task_dir / str(incumbent)
    winner = TabularPredictor.load(str(winner_path), verbosity=0)
    test_metrics = _evaluate(winner, test, spec)
    leaderboard = winner.leaderboard(test, silent=True).replace({np.nan: None}).head(12)
    leaderboard_rows = json.loads(leaderboard.to_json(orient="records"))
    sample = test.drop(columns=spec.label).head(5)
    predictions = winner.predict(sample).tolist()
    return {
        "spec": asdict(spec), "rows": len(frame), "features": frame.shape[1] - 1,
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "winner": incumbent, "test_metrics": test_metrics, "trials": trials,
        "leaderboard": leaderboard_rows, "sample_predictions": predictions,
        "model_path": str(winner_path.relative_to(artifact_dir)).replace("\\", "/"),
    }


def train_all(artifact_dir: Path | str = ROOT / "artifacts", time_limit: int = 15,
              max_candidates: int = 2) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {key: run_task(key, artifact_dir, time_limit, max_candidates) for key in TASKS}
    payload = {
        "project": "AutoML Research Console", "framework": "CRISP-DM",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "budget": {"seconds_per_candidate": time_limit, "candidates_per_task": max_candidates,
                   "cpu_limit": 2, "gpu": False},
        "methodology": {
            "search_policy": "greedy hill climbing over an ordered AutoGluon portfolio",
            "selection_data": "validation only", "holdout_used_for_search": False,
            "paper_alignment": ["heterogeneous model portfolio", "stacked/weighted ensembles", "anytime time budgets"],
        },
        "tasks": results, "total_duration_sec": time.perf_counter() - started,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_predictor(task: str, artifact_dir: Path | str = ROOT / "artifacts"):
    from autogluon.tabular import TabularPredictor
    base = Path(artifact_dir)
    metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    return TabularPredictor.load(str(base / metrics["tasks"][task]["model_path"]), verbosity=0)
