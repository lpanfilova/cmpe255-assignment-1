# Part 1: What's Cooking

This directory is for the AI-assisted **What's Cooking?** Kaggle data science project.

## Structure

- `data/` — project datasets
- `notebooks/` — exploratory and analysis notebooks
- `src/` — reusable source code
- `outputs/` — generated results and artifacts
- `screenshots/` — screenshots documenting the work
- `article.md` — project write-up

Data science models will be added later.

## Run the local demo

From `part-1-whats-cooking/`, install the packages and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pre-trained model is stored in `outputs/cuisine_classifier.joblib`. To recreate it from the training data, run:

```bash
python src/train_app_model.py
```
