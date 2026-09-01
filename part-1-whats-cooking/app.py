"""A tiny local Streamlit demonstration for the What's Cooking classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.model_utils import parse_ingredient_text


PART_DIR = Path(__file__).resolve().parent
MODEL_PATH = PART_DIR / "outputs" / "cuisine_classifier.joblib"
EXAMPLE_RECIPE = """chicken breasts
soy sauce
sesame oil
green onions
garlic
ginger"""


@st.cache_resource
def load_model():
    """Load the pre-trained artifact once per Streamlit server process."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="What's Cooking? Cuisine Predictor", page_icon="🍲")
st.title("What's Cooking? Cuisine Predictor")
st.write("Enter the ingredients of a recipe and the trained model will predict the most likely cuisine.")

if not MODEL_PATH.exists():
    st.error("The saved model is missing. Run `python src/train_app_model.py` first.")
    st.stop()

ingredients_text = st.text_area(
    "Ingredients (one per line or comma-separated)",
    placeholder=EXAMPLE_RECIPE,
    height=180,
)

if st.button("Predict Cuisine", type="primary"):
    raw_items = [item.strip() for item in ingredients_text.replace("\n", ",").split(",") if item.strip()]
    ingredients = parse_ingredient_text(ingredients_text)

    if not ingredients:
        st.warning("Enter at least one ingredient before predicting.")
    else:
        model = load_model()
        vectorizer = model.named_steps["vectorizer"]
        vocabulary = set(vectorizer.vocabulary_)
        unseen = [ingredient for ingredient in ingredients if ingredient not in vocabulary]

        probabilities = model.predict_proba([ingredients])[0]
        top_indices = probabilities.argsort()[-3:][::-1]
        top_predictions = pd.DataFrame(
            {
                "Cuisine": model.classes_[top_indices],
                "Probability": probabilities[top_indices],
            }
        )

        st.subheader(f"Predicted cuisine: {top_predictions.iloc[0]['Cuisine'].title()}")
        st.write("Top predicted cuisines")
        st.dataframe(
            top_predictions.style.format({"Probability": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )

        if len(ingredients) < len(raw_items):
            st.info("Duplicate ingredients were removed before prediction.")
        if unseen:
            st.info(
                f"{len(unseen)} ingredient phrase(s) were unseen during training and were ignored: "
                + ", ".join(unseen)
            )

        st.caption(
            "This is a demonstration model trained on the Kaggle What's Cooking dataset. "
            "Predictions are based only on ingredient phrases, not recipe quantities, methods, or IDs."
        )
