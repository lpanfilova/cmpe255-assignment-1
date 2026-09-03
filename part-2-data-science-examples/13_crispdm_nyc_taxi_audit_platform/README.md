# TaxiLens — CRISP-DM NYC Taxi Audit Platform

TaxiLens is a compact, locally runnable audit platform for a taxi trip-duration workflow. It makes the data contract, CRISP-DM decisions, model comparison, hyperparameter sensitivity, ablations, clustering, explainability, code seams, inference, and operating controls visible in one website.

The bundled data is a deterministic **synthetic teaching sample shaped like NYC TLC trips**, not real TLC evidence. This keeps setup fast and offline; replace `generate_sample()` with a reviewed TLC extract before drawing operational conclusions.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m taxi_audit.pipeline
python app.py
```

Open <http://127.0.0.1:5013>. The first page/API request also trains automatically if artifacts do not exist.

Example inference:

```bash
curl -X POST http://127.0.0.1:5013/api/v1/predict -H "Content-Type: application/json" -d '{"distance_miles":4.2,"passenger_count":2,"pickup_hour":17,"day_of_week":4,"temperature_f":71,"rain":0,"pickup_zone":"Midtown"}'
```

Other endpoints are `GET /api/v1/audit` and `GET /health`. With the server running, demonstrate concurrent load via `python load_test.py --requests 50 --workers 8`. The local development target is p95 below 500 ms; use a production WSGI server before treating this as a deployment benchmark.

## What is implemented

- Leakage-safe chronological 80/20 holdout and preprocessing contained in sklearn pipelines.
- Ridge, random forest, and gradient boosting comparison using MAE, RMSE, and R².
- Boosting depth sensitivity and four feature-group ablations.
- EDA, hourly congestion chart, K-means clustering visualization, and silhouette score.
- Holdout permutation importance plus per-request local what-if effects.
- Data-science audit gates for contract quality, leakage, baselines, XAI, reproducibility, and external validity.
- Versioned REST inference, validation errors, artifact SHA-256 fingerprint, drift signal, health endpoint, retraining policy, latency SLO, and dependency-free load demonstration.
- Auditor-oriented website with CRISP-DM narrative and highlighted implementation snippets.

On the deterministic 1,800-row sample, the exact champion and metrics are written to `artifacts/audit_report.json`; the dashboard always reads this generated evidence rather than hard-coding results.

## Test

```bash
pytest
```

The tests exercise data contracts, training and evaluation, clustering/ablation artifacts, persisted model inference and explanations, website rendering, audit/health routes, valid API inference, and invalid-request handling.
