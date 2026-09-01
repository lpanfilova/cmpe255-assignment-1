# query-validation

- Collection: `nimrodfisher`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: SQL query review for correctness, performance, and best practices. Activate when a query needs review before production use, shows unexpected results, or runs too slowly.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Loaded the passenger table into in-memory SQLite and reconciled SQL survival rate to pandas.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
