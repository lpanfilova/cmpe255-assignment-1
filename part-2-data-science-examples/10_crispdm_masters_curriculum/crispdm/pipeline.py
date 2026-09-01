"""Deterministic, laptop-sized implementations of the curriculum workflows."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "results.json"
RANDOM_STATE = 42
NUMERIC = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the dataset contract early."""
    frame = pd.read_csv(path)
    required = {"PassengerId", "Survived", *NUMERIC, *CATEGORICAL}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["PassengerId"].duplicated().any() or not frame["Survived"].isin([0, 1]).all():
        raise ValueError("PassengerId must be unique and Survived must be binary")
    return frame


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def data_understanding(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "survival_rate": round(float(frame.Survived.mean()), 4),
        "missing": {key: int(value) for key, value in frame.isna().sum().items() if value},
        "class_survival": {
            str(int(key)): round(float(value), 4)
            for key, value in frame.groupby("Pclass")["Survived"].mean().items()
        },
        "sex_survival": {
            str(key): round(float(value), 4) for key, value in frame.groupby("Sex")["Survived"].mean().items()
        },
    }


def clustering(frame: pd.DataFrame) -> dict:
    matrix = _preprocessor().fit_transform(frame[NUMERIC + CATEGORICAL])
    candidates = []
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(matrix)
        candidates.append((k, float(silhouette_score(matrix, labels))))
    best_k, best_score = max(candidates, key=lambda item: item[1])
    labels = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(matrix)
    profiled = frame.assign(cluster=labels).groupby("cluster").agg(
        size=("PassengerId", "size"), age=("Age", "median"), fare=("Fare", "median"),
        survival_rate=("Survived", "mean"), female_share=("Sex", lambda values: (values == "female").mean())
    )
    profiles = []
    for cluster_id, row in profiled.reset_index().iterrows():
        profiles.append({
            "cluster": int(row["cluster"]), "size": int(row["size"]),
            "median_age": None if pd.isna(row["age"]) else round(float(row["age"]), 1),
            "median_fare": round(float(row["fare"]), 2),
            "survival_rate": round(float(row["survival_rate"]), 3),
            "female_share": round(float(row["female_share"]), 3),
        })
    return {"best_k": best_k, "silhouette": round(best_score, 3), "profiles": profiles,
            "candidate_scores": {str(k): round(score, 3) for k, score in candidates}}


def anomaly_detection(frame: pd.DataFrame) -> dict:
    numeric = frame[NUMERIC].copy()
    numeric = numeric.fillna(numeric.median())
    detector = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    flags = detector.fit_predict(StandardScaler().fit_transform(numeric)) == -1
    scores = -detector.score_samples(StandardScaler().fit_transform(numeric))
    ranked = frame.assign(anomaly_score=scores, is_anomaly=flags).sort_values("anomaly_score", ascending=False).head(8)
    records = [{
        "passenger_id": int(row.PassengerId), "name": row.Name, "age": None if pd.isna(row.Age) else float(row.Age),
        "fare": round(float(row.Fare), 2), "family_size": int(row.SibSp + row.Parch + 1),
        "score": round(float(row.anomaly_score), 3)
    } for row in ranked.itertuples()]
    return {"count": int(flags.sum()), "rate": round(float(flags.mean()), 3), "top": records}


def supervised_learning(frame: pd.DataFrame) -> dict:
    features = frame[NUMERIC + CATEGORICAL]
    target = frame["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=RANDOM_STATE
    )
    estimators = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=3, random_state=RANDOM_STATE),
    }
    metrics = {}
    for name, estimator in estimators.items():
        model = Pipeline([("prepare", _preprocessor()), ("model", estimator)])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "f1": round(float(f1_score(y_test, predictions)), 3),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
        }
    best = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    return {"split": {"train": len(x_train), "test": len(x_test)}, "models": metrics, "best_model": best}


def _transactions(frame: pd.DataFrame) -> list[set[str]]:
    age = pd.cut(frame.Age, [-np.inf, 15, 30, 50, np.inf], labels=["child", "young", "middle", "older"])
    fare = pd.qcut(frame.Fare.rank(method="first"), 3, labels=["low", "mid", "high"])
    transactions = []
    for i, row in frame.iterrows():
        transactions.append({f"sex={row.Sex}", f"class={row.Pclass}", f"embarked={row.Embarked}",
                             f"age={age.iloc[i] if pd.notna(age.iloc[i]) else 'unknown'}", f"fare={fare.iloc[i]}",
                             f"outcome={'survived' if row.Survived else 'died'}"})
    return transactions


def association_rules(frame: pd.DataFrame, min_support: float = 0.08, min_confidence: float = 0.65) -> dict:
    transactions = _transactions(frame)
    counts: dict[frozenset[str], int] = {}
    for transaction in transactions:
        for size in (1, 2, 3):
            for itemset in combinations(sorted(transaction), size):
                key = frozenset(itemset)
                counts[key] = counts.get(key, 0) + 1
    supports = {key: value / len(transactions) for key, value in counts.items() if value / len(transactions) >= min_support}
    rules = []
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = itemset - {consequent}
            antecedent_support = supports.get(frozenset(antecedent))
            consequent_support = supports.get(frozenset({consequent}))
            if not antecedent_support or not consequent_support:
                continue
            confidence = support / antecedent_support
            lift = confidence / consequent_support
            if confidence >= min_confidence and lift > 1.05:
                rules.append({"if": sorted(antecedent), "then": consequent, "support": round(support, 3),
                              "confidence": round(confidence, 3), "lift": round(lift, 3)})
    rules.sort(key=lambda rule: (rule["lift"], rule["support"]), reverse=True)
    return {"min_support": min_support, "min_confidence": min_confidence, "rules": rules[:12]}


def _tokens(row) -> set[str]:
    age_band = "unknown" if pd.isna(row.Age) else str(int(row.Age // 10) * 10)
    return {f"sex:{row.Sex}", f"class:{row.Pclass}", f"embarked:{row.Embarked}",
            f"age_decade:{age_band}", f"family:{min(row.SibSp + row.Parch, 3)}"}


def _hash(token: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{token}".encode(), digest_size=8).digest(), "big")


def lsh_search(frame: pd.DataFrame, query_index: int = 0, permutations: int = 48, bands: int = 12) -> dict:
    """MinHash LSH: band signatures narrow candidates before exact Jaccard ranking."""
    token_sets = [_tokens(row) for row in frame.itertuples()]
    signatures = np.array([[min(_hash(token, seed) for token in tokens) for seed in range(permutations)] for tokens in token_sets])
    rows_per_band = permutations // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = (band, tuple(signature[band * rows_per_band:(band + 1) * rows_per_band]))
            buckets.setdefault(key, []).append(index)
    candidates = set()
    for band in range(bands):
        key = (band, tuple(signatures[query_index, band * rows_per_band:(band + 1) * rows_per_band]))
        candidates.update(buckets[key])
    candidates.discard(query_index)
    query_tokens = token_sets[query_index]
    scored = sorted(((len(query_tokens & token_sets[i]) / len(query_tokens | token_sets[i]), i) for i in candidates), reverse=True)[:5]
    neighbors = [{"passenger_id": int(frame.iloc[i].PassengerId), "name": frame.iloc[i].Name,
                  "similarity": round(float(score), 3), "tokens": sorted(token_sets[i])} for score, i in scored]
    return {"query": {"passenger_id": int(frame.iloc[query_index].PassengerId), "name": frame.iloc[query_index].Name,
                      "tokens": sorted(query_tokens)}, "candidates_examined": len(candidates),
            "total_rows": len(frame) - 1, "reduction": round(1 - len(candidates) / (len(frame) - 1), 3), "neighbors": neighbors}


def run_workflows(data_path: Path = DATA_PATH, output_path: Path | None = ARTIFACT_PATH) -> dict:
    frame = load_data(data_path)
    result = {"metadata": {"dataset": "Kaggle Titanic train set", "random_state": RANDOM_STATE},
              "understanding": data_understanding(frame), "clustering": clustering(frame),
              "anomalies": anomaly_detection(frame), "supervised": supervised_learning(frame),
              "associations": association_rules(frame), "lsh": lsh_search(frame)}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    results = run_workflows()
    print(f"Wrote {ARTIFACT_PATH} with {results['understanding']['rows']} analyzed rows")

