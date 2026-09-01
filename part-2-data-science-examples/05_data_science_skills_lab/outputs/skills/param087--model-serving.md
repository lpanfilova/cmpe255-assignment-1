# model-serving

- Collection: `param087`
- Dataset: Titanic: Machine Learning from Disaster
- Skill intent: Use when deploying a trained model behind an API. Covers FastAPI inference services, loading artifacts safely, request validation, batching, ONNX/quantization for speed, health checks, and monitoring.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.
- The Flask /api/predict endpoint validates a seven-field passenger request and returns a probability.
- The /api/health endpoint exposes artifact readiness without leaking internals.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
