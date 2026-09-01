"""Fit the final app model once and save it for Streamlit to load."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PART_DIR = Path(__file__).resolve().parents[1]
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from src.model_utils import build_model, normalize_ingredients


def main() -> None:
    train_path = PART_DIR / "data" / "train.json"
    model_path = PART_DIR / "outputs" / "cuisine_classifier.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open(encoding="utf-8") as file:
        train = pd.DataFrame(json.load(file))

    features = train["ingredients"].apply(normalize_ingredients).tolist()
    model = build_model()
    model.fit(features, train["cuisine"])
    joblib.dump(model, model_path)

    print(f"Trained on {len(train):,} recipes.")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
