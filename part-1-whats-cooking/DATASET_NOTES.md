# Dataset Notes, Modeling Plan, and Final Implementation

## Source files inspected

The raw dataset files are:

- `data/train.json`
- `data/test.json`

`data/sample_submission.csv` has the expected columns: `id,cuisine`.

## Dataset summary

| Split | Records | Fields |
| --- | ---: | --- |
| Training | 39,774 | `id`, `cuisine`, `ingredients` |
| Test | 9,944 | `id`, `ingredients` |

Each record represents one recipe. `id` is an integer identifier, `ingredients` is a list of ingredient strings, and `cuisine` is the target label available only in the training set. This is a **20-class multiclass classification** problem.

No keys are missing, no field values are null, and no empty ingredient lists were found. All ingredient entries are non-empty strings. IDs are unique within each split, with no train/test ID overlap.

## Cuisine classes and distribution

The class distribution is imbalanced: Italian is the largest class (19.71%) and Brazilian is the smallest (1.17%). A majority-class baseline that always predicts Italian would achieve 19.71% training accuracy.

| Cuisine | Recipes | Share |
| --- | ---: | ---: |
| italian | 7,838 | 19.71% |
| mexican | 6,438 | 16.19% |
| southern_us | 4,320 | 10.86% |
| indian | 3,003 | 7.55% |
| chinese | 2,673 | 6.72% |
| french | 2,646 | 6.65% |
| cajun_creole | 1,546 | 3.89% |
| thai | 1,539 | 3.87% |
| japanese | 1,423 | 3.58% |
| greek | 1,175 | 2.95% |
| spanish | 989 | 2.49% |
| korean | 830 | 2.09% |
| vietnamese | 825 | 2.07% |
| moroccan | 821 | 2.06% |
| british | 804 | 2.02% |
| filipino | 755 | 1.90% |
| irish | 667 | 1.68% |
| jamaican | 526 | 1.32% |
| russian | 489 | 1.23% |
| brazilian | 467 | 1.17% |

## Ingredients

There are 6,714 distinct ingredient strings in training data. The test set contains 4,484 distinct strings, including 423 strings not seen in training; these unseen ingredients are naturally ignored by a vocabulary learned only from training data.

Recipes contain a moderate number of ingredients:

| Split | Minimum | Q1 | Median | Mean | Q3 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 1 | 8 | 10 | 10.77 | 13 | 65 |
| Test | 1 | 8 | 10 | 10.80 | 13 | 50 |

In training, 42.68% of recipes have 6–10 ingredients and 33.74% have 11–15. The most frequent list length is 9 ingredients (3,753 recipes). The long right tail should be described in EDA, but it does not by itself require removal of long recipes.

The 20 most common ingredients across train and test are:

1. salt (22,534)
2. onions (10,008)
3. olive oil (9,889)
4. water (9,293)
5. garlic (9,171)
6. sugar (8,064)
7. garlic cloves (7,772)
8. butter (6,078)
9. ground black pepper (5,990)
10. all-purpose flour (5,816)
11. vegetable oil (5,516)
12. pepper (5,508)
13. eggs (4,262)
14. soy sauce (4,120)
15. kosher salt (3,930)
16. green onions (3,817)
17. tomatoes (3,812)
18. large eggs (3,700)
19. carrots (3,542)
20. unsalted butter (3,474)

### Example training recipes

- **Greek** (ID 10259): romaine lettuce, black olives, grape tomatoes, garlic, pepper, purple onion, seasoning, garbanzo beans, feta cheese crumbles.
- **Southern US** (ID 25693): plain flour, ground pepper, salt, tomatoes, ground black pepper, thyme, eggs, green tomatoes, yellow corn meal, milk, vegetable oil.
- **Filipino** (ID 20130): eggs, pepper, salt, mayonaise, cooking oil, green chilies, grilled chicken breasts, garlic powder, yellow onion, soy sauce, butter, chicken livers.

## Data-quality findings and cautions

- The schema is complete and clean with respect to nulls and IDs.
- There are 100 extra training rows that duplicate an **exact ordered** ingredient list, forming 96 duplicate groups. When ingredient order is ignored, there are 531 extra rows with the same ingredient set.
- Fourteen order-insensitive duplicate ingredient sets have conflicting cuisine labels. For example, `chili oil`, `rice vinegar`, and `soy sauce` occur under both Chinese and Japanese. Do not automatically deduplicate recipes or assume an identical ingredient set always has one correct cuisine.
- There are 58 test recipes with an exact ordered ingredient-list match in training, or 280 when ingredient order is ignored. This makes ordinary random validation slightly optimistic if duplicate sets are split between folds. Group identical normalized ingredient sets during local validation.
- Ingredient text is inconsistent in ways that may fragment the vocabulary: `garlic` versus `garlic cloves`, `butter` versus `unsalted butter`, branded products, spelling errors such as `mayonaise`, and capitalization variants such as `Pace Picante Sauce` versus `pace picante sauce`. There are also 26 repeated ingredient entries within training recipes.
- Do not use `id` as a feature. It is only an identifier and could create spurious validation patterns.
- Do not fit text normalization, the vectorizer vocabulary, feature selection, or hyperparameter choices on validation/test information. Fit all transformations within each training fold, then refit on all training data only after selecting the approach.

## Initial modeling strategy

This section records the modeling plan established before the experiment was run. It is retained to document the original rationale and should not be read as a claim that its originally recommended candidate became the final selected model.

Use the full ingredient phrase as the fundamental feature, rather than splitting phrases into individual words. A binary multi-hot representation is intuitive: each ingredient is a feature with value 1 when it appears in the recipe and 0 otherwise. Lowercase and trim ingredient strings consistently, but avoid aggressive stemming, stop-word removal, or manually collapsing related ingredients in the first version; phrases such as `soy sauce` and `olive oil` carry useful meaning.

Compare these models using the same features and validation folds:

1. **Baseline:** majority-class predictor, then a simple `MultinomialNB` model with binary ingredient features.
2. **Interpretable linear model:** multinomial Logistic Regression with a modest regularization search.
3. **Recommended final candidate:** Linear SVM (`LinearSVC`) with binary ingredient features and a small search over `C` (for example, 0.25, 0.5, 1, and 2). This is usually a strong, efficient choice for sparse high-dimensional recipe data and remains straightforward to explain.

Use 5-fold **stratified group cross-validation**, grouping by a normalized, order-insensitive ingredient list. If the installed scikit-learn version does not provide `StratifiedGroupKFold`, use a grouped split and report the resulting class-balance limitation. Compare unweighted and `class_weight="balanced"` versions of the linear models. Choose the final model by mean validation accuracy, using macro F1 and per-class recall as secondary checks so that performance on small cuisines is not hidden by the large Italian and Mexican classes.

The Kaggle-style submission should contain one predicted cuisine for each test `id`; therefore accuracy is the primary competition-aligned metric. Macro F1, a confusion matrix, and per-class recall should accompany the assignment discussion because the labels are imbalanced.

### Implementation deviations from the initial plan

- The initial plan called for 5-fold `StratifiedGroupKFold`. The completed experiment used 3-fold `StratifiedGroupKFold` as a practical runtime compromise while preserving grouped, stratified validation.
- The planned broader `C` and class-weight comparisons were intentionally not expanded into a large hyperparameter search. The completed experiment compared the four documented baseline/model configurations with the same grouped folds and random state.

## Canonical environment

The canonical project environment is Conda environment `cmpe255-a1`:

| Package | Version |
| --- | --- |
| Python | 3.11.16 |
| NumPy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |

The original notebook was produced against an older environment. After standardizing on Python 3.11 and scikit-learn 1.9, a compatibility audit identified version-sensitive Logistic Regression and `LinearSVC` behavior. The complete notebook was therefore rerun in the canonical environment, and all final artifacts were regenerated.

## Final model comparison

These are the canonical 3-fold grouped-validation results:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.7453 | 0.6527 |
| Linear SVM | 0.7426 | 0.6552 |
| Logistic Regression | 0.7405 | 0.6660 |
| Majority-class baseline | 0.1971 | 0.0165 |

Accuracy remained the primary model-selection metric. `MultinomialNB(alpha=0.5)` achieved the highest accuracy and was selected as the final model. Logistic Regression achieved the highest macro F1, illustrating the tradeoff between overall accuracy and more balanced per-class performance.

## Final selected model and artifacts

The selected pipeline uses a binary `CountVectorizer` representation of full normalized ingredient phrases. Ingredients are lowercased, trimmed, and de-duplicated within each recipe; phrases are not split into individual words. Recipe `id` is excluded. The final `MultinomialNB(alpha=0.5)` model was fitted on all training recipes only after model comparison.

This final modeling approach regenerated:

- `outputs/submission.csv` with 9,944 Kaggle-format predictions
- `outputs/cuisine_classifier.joblib`
- evaluation and interpretation artifacts in `outputs/`

The same fitted modeling approach is used by the Streamlit demonstration app.

## Useful EDA visualizations

1. A sorted horizontal bar chart of cuisine counts and percentages.
2. A histogram of ingredient-list lengths, with median and quartiles marked; optionally compare train and test distributions.
3. A horizontal bar chart of the 20 most common ingredients overall.
4. Small-multiple or heatmap view of the most distinctive ingredients per cuisine (for example, the highest within-cuisine prevalence or TF-IDF-style association), rather than only globally common ingredients such as salt.
5. A cuisine-by-ingredient-count box plot or violin plot to compare recipe complexity across cuisines.
6. After model evaluation begins, a normalized confusion matrix and a per-class precision/recall/F1 chart.

## Current project status

- EDA and model comparison are complete.
- The canonical notebook executes successfully from top to bottom.
- A Kaggle-format submission has been generated.
- A small Streamlit prediction app has been created and smoke-tested.
