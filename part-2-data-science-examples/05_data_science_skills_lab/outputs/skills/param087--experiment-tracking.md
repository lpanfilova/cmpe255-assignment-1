# experiment-tracking

- Collection: `param087`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Use when running ML experiments that need to be compared, reproduced, or shared. Covers MLflow/Weights & Biases logging, what to track, run organization, and model registry basics.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
