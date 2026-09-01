# llm-finetuning

- Collection: `param087`
- Dataset: Titanic (row-to-instruction formatting smoke demo)
- Skill intent: Use when fine-tuning a large language model. Covers choosing full vs LoRA/QLoRA, dataset formatting, the transformers/PEFT/TRL stack, key hyperparameters, and evaluating fine-tunes without overfitting.

## Demonstration evidence

- Validated 891 rows and 12 columns at passenger grain.
- Leakage-safe held-out ROC AUC: 0.840; balanced accuracy: 0.757.
- Training uses a fixed seed, stratification, train-only preprocessing, and {'model__C': 0.25, 'model__class_weight': None}.
- Formatted passenger rows as instruction/response examples; deliberately skipped large-model training.
- Defined holdout, privacy review, and task-accuracy gates before any real fine-tune.

## Guardrail

This compact lab demonstrates the workflow contract. It does not claim production readiness or causal inference.
