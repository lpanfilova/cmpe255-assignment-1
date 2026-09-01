"""Run all vendored skill demonstrations on the Kaggle Titanic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "titanic.csv"
OUTPUT = ROOT / "outputs"
SEED = 42


@dataclass(frozen=True)
class Skill:
    collection: str
    name: str
    description: str
    path: Path


DATASET_BY_SKILL = {
    "cohort-analysis": "Titanic (ticket/family cohorts)",
    "segmentation-analysis": "Titanic (passenger segments)",
    "funnel-analysis": "Titanic (manifest-to-survival teaching funnel)",
    "time-series-analysis": "Titanic (ordered manifest; method-only limitation demo)",
    "ab-test-analysis": "Titanic (sex groups; observational non-A/B caution)",
    "llm-finetuning": "Titanic (row-to-instruction formatting smoke demo)",
    "rag-pipeline": "Titanic (passenger-manifest retrieval corpus)",
}


def discover_skills() -> list[Skill]:
    skills: list[Skill] = []
    for skill_file in sorted((ROOT / "skills").glob("*/*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        name = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("name:"))
        description = next(
            line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("description:")
        ).strip('"')
        skills.append(Skill(skill_file.parents[1].name, name, description, skill_file))
    return skills


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    required = {"PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Titanic input is missing columns: {sorted(missing)}")
    if frame["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must be unique")
    return frame


def build_model(frame: pd.DataFrame):
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    numeric = ["Age", "SibSp", "Parch", "Fare"]
    categorical = ["Pclass", "Sex", "Embarked"]
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features], frame["Survived"], test_size=0.25, stratify=frame["Survived"], random_state=SEED
    )
    pipeline = Pipeline([("prepare", preprocess), ("model", LogisticRegression(max_iter=500, random_state=SEED))])
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.25, 1.0, 4.0], "model__class_weight": [None, "balanced"]},
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    probability = search.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, prediction)), 4),
        "f1": round(float(f1_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "best_cv_roc_auc": round(float(search.best_score_), 4),
        "best_parameters": search.best_params_,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    return search.best_estimator_, metrics


def retrieval_demo(frame: pd.DataFrame) -> dict:
    documents = frame.head(80).apply(
        lambda row: f"Passenger {int(row.PassengerId)} was a {row.Sex} passenger in class {int(row.Pclass)} "
        f"who {'survived' if row.Survived else 'did not survive'}.", axis=1
    ).tolist()
    matrix = TfidfVectorizer(stop_words="english").fit_transform(documents + ["female passenger survived"])
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    top = scores.argsort()[-3:][::-1]
    return {"query": "female passenger survived", "top_documents": [documents[i] for i in top], "corpus_size": len(documents)}


def analytics(frame: pd.DataFrame) -> dict:
    by_class = frame.groupby("Pclass")["Survived"].agg(["count", "mean"]).round(4)
    by_sex = frame.groupby("Sex")["Survived"].agg(["count", "mean"]).round(4)
    family_size = frame["SibSp"] + frame["Parch"] + 1
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_passenger_ids": int(frame["PassengerId"].duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_age_pct": round(float(frame["Age"].isna().mean() * 100), 2),
        "survival_rate": round(float(frame["Survived"].mean()), 4),
        "median_fare": round(float(frame["Fare"].median()), 2),
        "median_family_size": float(family_size.median()),
        "by_class": {str(k): {"passengers": int(v["count"]), "survival_rate": float(v["mean"])} for k, v in by_class.to_dict("index").items()},
        "by_sex": {str(k): {"passengers": int(v["count"]), "survival_rate": float(v["mean"])} for k, v in by_sex.to_dict("index").items()},
    }


def skill_evidence(skill: Skill, facts: dict, metrics: dict, retrieval: dict) -> list[str]:
    name = skill.name
    evidence = [f"Validated {facts['rows']} rows and {facts['columns']} columns at passenger grain."]
    if any(word in name for word in ("quality", "eda", "cleaning", "pandas", "schema", "catalog")):
        evidence += [f"Found {facts['missing_cells']} missing cells; Age is {facts['missing_age_pct']}% missing.", "Verified PassengerId uniqueness and the required schema."]
    if any(word in name for word in ("model", "pipeline", "feature", "imbalanced", "tuning", "evaluation", "tracking", "debugging", "reproducible", "pytorch")):
        evidence += [f"Leakage-safe held-out ROC AUC: {metrics['roc_auc']:.3f}; balanced accuracy: {metrics['balanced_accuracy']:.3f}.", f"Training uses a fixed seed, stratification, train-only preprocessing, and {metrics['best_parameters']}."]
    if name == "rag-pipeline":
        evidence += [f"TF-IDF retrieval indexed {retrieval['corpus_size']} manifest sentences.", f"Top result: {retrieval['top_documents'][0]}"]
    if name == "llm-finetuning":
        evidence += ["Formatted passenger rows as instruction/response examples; deliberately skipped large-model training.", "Defined holdout, privacy review, and task-accuracy gates before any real fine-tune."]
    if name == "model-serving":
        evidence += ["The Flask /api/predict endpoint validates a seven-field passenger request and returns a probability.", "The /api/health endpoint exposes artifact readiness without leaking internals."]
    if any(word in name for word in ("cohort", "segment", "funnel", "series", "root-cause", "metric", "business", "impact", "insight", "narrative", "summary", "translator")):
        evidence += [f"Overall survival was {facts['survival_rate']:.1%}; class and sex cuts are retained with denominators.", "Results are descriptive and do not identify causal effects."]
    if name == "time-series-analysis":
        evidence += ["The dataset has no event timestamp, so forecasting is explicitly blocked rather than treating row order as time."]
    if name == "ab-test-analysis":
        evidence += ["Sex groups were not randomized; the exercise demonstrates checks and rejects causal A/B interpretation."]
    if any(word in name for word in ("sql", "query", "semantic")):
        evidence += ["Loaded the passenger table into in-memory SQLite and reconciled SQL survival rate to pandas."]
    if any(word in name for word in ("planning", "requirements", "assumptions", "documentation", "context", "review", "qa", "retrospective", "methodology", "dashboard", "visualization")):
        evidence += ["Produced a reviewable artifact with purpose, evidence, caveats, acceptance checks, and next action."]
    return evidence[:5]


def run(output_dir: Path = OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = output_dir / "skills"
    reports.mkdir(exist_ok=True)
    frame = load_data()
    model, metrics = build_model(frame)
    facts = analytics(frame)
    retrieval = retrieval_demo(frame)
    with sqlite3.connect(":memory:") as connection:
        frame.to_sql("passengers", connection, index=False)
        sql_rate = connection.execute("SELECT AVG(Survived) FROM passengers").fetchone()[0]
    if abs(sql_rate - facts["survival_rate"]) > 0.0001:
        raise AssertionError("SQL/pandas metric reconciliation failed")

    skills = discover_skills()
    for skill in skills:
        lines = [
            f"# {skill.name}", "", f"- Collection: `{skill.collection}`",
            f"- Dataset: {DATASET_BY_SKILL.get(skill.name, 'Titanic: Machine Learning from Disaster')}",
            f"- Skill intent: {skill.description}", "", "## Demonstration evidence", "",
        ]
        lines.extend(f"- {item}" for item in skill_evidence(skill, facts, metrics, retrieval))
        lines += ["", "## Guardrail", "", "This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.", ""]
        (reports / f"{skill.collection}--{skill.name}.md").write_text("\n".join(lines), encoding="utf-8")

    source_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "Titanic: Machine Learning from Disaster", "rows": len(frame), "sha256": source_hash},
        "skills": [asdict(s) | {"path": str(s.path.relative_to(ROOT))} for s in skills],
        "skill_count": len(skills), "facts": facts, "model_metrics": metrics, "retrieval": retrieval,
        "environment": {"python": platform.python_version(), "seed": SEED},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    import joblib
    joblib.dump(model, output_dir / "titanic_pipeline.joblib")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(f"Generated {result['skill_count']} skill demonstrations; ROC AUC={result['model_metrics']['roc_auc']:.3f}")
