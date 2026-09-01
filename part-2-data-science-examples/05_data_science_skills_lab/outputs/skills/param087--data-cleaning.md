# data-cleaning

- Collection: `param087`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Use when preparing raw data for modeling — handling missing values, duplicates, inconsistent types, outliers, and bad categorical values. Emphasizes fitting all imputation on train-only to avoid leakage.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Found 866 missing cells; Age is 19.87% missing.
- Verified PassengerId uniqueness and the required schema.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
