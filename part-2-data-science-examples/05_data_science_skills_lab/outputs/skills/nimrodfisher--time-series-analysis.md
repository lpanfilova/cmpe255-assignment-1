# time-series-analysis

- Collection: `nimrodfisher`
- Dataset: Titanic (ordered manifest; method-only limitation demo)
- Skill intent: Temporal pattern detection and forecasting. Use when analyzing trends over time, detecting seasonality, identifying anomalies in time series, or building simple forecasting models for planning.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Overall survival was 38.4%; class and sex cuts are retained with denominators.
- Results are descriptive and do not identify causal effects.
- The dataset has no event timestamp, so forecasting is explicitly blocked rather than treating row order as time.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
