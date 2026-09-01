# semantic-model-builder

- Collection: `nimrodfisher`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Build structured semantic layer documentation for metrics, dimensions, and entities. Activate when you need to define a business metric, document a data model, or create YAML definitions compatible with dbt Semantic Layer or similar frameworks.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.
- Loaded the passenger table into in-memory SQLite and reconciled SQL survival rate to pandas.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
