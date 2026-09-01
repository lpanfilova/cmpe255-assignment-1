# sql-to-business-logic

- Collection: `nimrodfisher`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Translate SQL queries into plain language business logic. Use when documenting queries, explaining analysis to non-technical stakeholders, code reviewing for correctness, or building a query catalog.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Overall survival was 38.4%; class and sex cuts are retained with denominators.
- Results are descriptive and do not identify causal effects.
- Loaded the passenger table into in-memory SQLite and reconciled SQL survival rate to pandas.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
