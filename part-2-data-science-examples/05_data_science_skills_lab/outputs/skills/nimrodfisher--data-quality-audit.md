# data-quality-audit

- Collection: `nimrodfisher`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Comprehensive data quality assessment against business rules, schema constraints, and freshness expectations. Activate when validating data pipeline outputs before production use, auditing a dataset against defined business rules, or producing a quality scorecard for a data asset.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Found 866 missing cells; Age is 19.87% missing.
- Verified PassengerId uniqueness and the required schema.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
