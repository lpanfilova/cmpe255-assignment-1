# rag-pipeline

- Collection: `param087`
- Dataset: Titanic (passenger-manifest retrieval corpus)
- Skill intent: Use when building retrieval-augmented generation. Covers chunking strategy, embedding choice, vector stores, hybrid + reranking retrieval, prompt assembly, and evaluating retrieval and answer quality.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.
- TF-IDF retrieval indexed 80 manifest sentences.
- Top result: Passenger 2 was a female passenger in class 1 who survived.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
