# imbalanced-data

- Collection: `param087`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Use when the target is rare (fraud, churn, disease, anomalies). Covers correct metrics, resampling (SMOTE/undersampling), class weights, threshold tuning, and avoiding the accuracy trap and resampling leakage.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
