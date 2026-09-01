"""Preprocessing and model construction shared by training and the Streamlit app."""

from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def normalize_ingredients(ingredients: Iterable[str]) -> List[str]:
    """Lowercase, trim, and de-duplicate full ingredient phrases.

    Phrases are deliberately not split into word tokens: ``olive oil`` and
    ``soy sauce`` remain individual model features, matching the notebook.
    """
    return sorted(
        {
            ingredient.strip().lower()
            for ingredient in ingredients
            if ingredient and ingredient.strip()
        }
    )


def parse_ingredient_text(text: str) -> List[str]:
    """Parse comma- or newline-separated ingredient phrases for the demo UI."""
    return normalize_ingredients(text.replace("\n", ",").split(","))


def identity(document):
    """Return a pre-tokenized ingredient list unchanged for CountVectorizer."""
    return document


def build_model() -> Pipeline:
    """Create the validation-selected MultinomialNB pipeline from the notebook."""

    return Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    analyzer=identity,
                    lowercase=False,
                    binary=True,
                    dtype=np.int8,
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.5),
            ),
        ]
    )
