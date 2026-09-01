# sklearn-pipelines

- Collection: `param087`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Use when building scikit-learn models that must not leak preprocessing. Covers Pipeline, ColumnTransformer, custom transformers, and combining preprocessing with cross-validation correctly.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
