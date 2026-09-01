# AI Coding Prompts

## Prompt 1 — Repository Setup

I am working on CMPE 255 Assignment 1 for a graduate Data Mining course.

Please scaffold this repository only. Do not implement any data science models yet.

Create a clean structure with:

part-1-whats-cooking/
  data/
  notebooks/
  src/
  outputs/
  screenshots/
  article.md
  README.md

part-2-data-science-examples/
  README.md

Update the root README.md with:
- title: CMPE 255 - Assignment 1
- a Part 1 section for an AI-assisted What's Cooking Kaggle data science project
- a Part 2 section for reproducing 14 professor's provided data science experiments

Keep it simple and appropriate for a university assignment.
Before finishing, explain what files you created and why.
Do not commit or push anything to GitHub.

## Prompt 2 — Dataset Inspection and Planning

We are now starting Part 1 of my CMPE 255 Assignment 1.

The project uses the Kaggle "What's Cooking?" dataset located in:
part-1-whats-cooking/data/

For this step, DO NOT train models and DO NOT build the application yet.

Your job is to inspect the dataset and propose a data science plan.

Please:

1. Inspect train.json and test.json.
2. Report:
   - number of training and test records
   - available fields
   - number of cuisine classes
   - class distribution
   - whether there are missing/null values
   - duplicate recipes if any
   - distribution of number of ingredients per recipe
   - most common ingredients overall
   - several example recipes
3. Identify any data-quality issues or possible modeling pitfalls.
4. Specifically think about:
   - whether this is a multiclass classification problem
   - appropriate ways to represent ingredient lists
   - potential target leakage
   - appropriate evaluation metrics
5. Propose a simple but strong modeling strategy appropriate for a graduate Data Mining assignment.

I would prefer something understandable rather than unnecessarily complicated.

Potential models may include:
- a simple baseline
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Do not assume these are necessarily the best choices; explain what you recommend.

6. Propose useful EDA visualizations.
7. Create:
   part-1-whats-cooking/DATASET_NOTES.md

The file should summarize what you actually found in the dataset and the proposed plan.

Do not modify the raw dataset.
Do not commit or push anything.
Before making any changes, inspect the existing repository structure.

## Prompt 3 - Model Training and Evaluation

I reviewed DATASET_NOTES.md and approve the overall modeling plan.

Now implement the core end-to-end data science experiment for Part 1.

Important constraints:
- Keep the implementation understandable and suitable for a graduate Data Mining assignment.
- Do not over-engineer.
- Do not build the web application yet.
- Do not commit or push anything.
- Never modify the raw files in data/.

Please create a reproducible Jupyter notebook in:
part-1-whats-cooking/notebooks/whats_cooking_analysis.ipynb

The notebook should tell a coherent data science story and be runnable from top to bottom.

Include:
1. Problem definition and load train/test
2. Dataset overview: dimensions, classes, missing values, duplicates, ingredient-count summary
3. EDA with saved plots in outputs/: class distribution, ingredient-count distribution, most common ingredients, one useful cuisine/ingredient visualization
4. Feature representation using normalized ingredient phrases; explain why ID is excluded and avoid leakage
5. Compare majority baseline, MultinomialNB, Logistic Regression, Linear SVM
6. Use sensible validation based on DATASET_NOTES.md; prefer StratifiedGroupKFold if clean, otherwise clearly explained stratified validation
7. Report accuracy, macro F1, per-class results, normalized confusion matrix
8. Select/refit best model and generate outputs/submission.csv
9. Interpretation: easiest/hardest cuisines, common confusions, influential/distinctive ingredients if practical, limitations
10. Use clear markdown cells and verify all cells run successfully

At the end summarize model results, best model, files generated, and methodological deviations.