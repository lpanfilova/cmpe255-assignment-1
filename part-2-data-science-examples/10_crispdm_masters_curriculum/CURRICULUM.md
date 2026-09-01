# CRISP-DM Master's Curriculum: Titanic as a Multi-Lens Case Study

## Learning contract

CRISP-DM is a loop, not a waterfall. Its six phases are **business understanding, data understanding, data preparation, modeling, evaluation, and deployment**. A mature project can move backward whenever evidence invalidates an assumption. By the end, a learner should be able to match a question to a method, protect evaluation from leakage, interpret metrics, and state what the evidence cannot establish.

The case question is: **Which observed passenger profiles were associated with survival, and how do different analytical methods expose complementary structure?** Prediction is evaluated by held-out ROC-AUC, accuracy, and F1. Descriptive methods are judged by internal quality and interpretability. This is historical observational data: every result is associational.

## 1. Business understanding

The Kaggle task predicts `Survived`. Our broader teaching goal adds five questions:

1. What recurring passenger profiles exist without using the outcome?
2. Which numeric profiles are unusual enough to review?
3. How accurately can a small model rank survival probability?
4. Which discretized attributes co-occur more than chance predicts?
5. How can we retrieve similar profiles without comparing every row?

**Phase gate:** the target, users, decision, metrics, constraints, and non-goals are explicit. A model with good AUC still fails this gate if its intended use is undefined.

## 2. Data understanding

The training table contains 891 unique passengers and 12 columns. `Survived` is binary. The audit checks schema, identifier uniqueness, label validity, missing values, cardinality, distributions, and grouped outcome rates. Missingness is itself informative: 177 ages, 687 cabins, and 2 embarkation ports are absent.

EDA is hypothesis generation, not hypothesis proof. The observed survival rates by sex and passenger class suggest predictive signal, but they mix access, policy, demographics, and other unobserved mechanisms.

**Phase gate:** grain and label are understood; duplicates and missingness are quantified; suspicious patterns have follow-up checks; no target-derived feature enters exploratory unsupervised inputs.

## 3. Data preparation

Numeric features are `Age`, `Fare`, `SibSp`, and `Parch`. Categorical features are `Pclass`, `Sex`, and `Embarked`.

- Numeric missing values receive the median; scaling maps features to comparable magnitudes.
- Categorical missing values receive the most frequent category; one-hot encoding avoids a false numeric order.
- For supervised learning, transformations live inside a scikit-learn `Pipeline` and are fitted only on training data. Fitting imputation or scaling before the split would leak test-distribution information.
- Association mining discretizes age and fare because itemset methods consume transactions.
- LSH converts each passenger into five interpretable tokens. The target is excluded.

**Phase gate:** every transformation is deterministic, justified, train/test-safe, and traceable to a model's data requirements.

## 4A. Clustering

K-means minimizes within-cluster squared distance:

\[
\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2.
\]

We test `k=2..5` using the silhouette coefficient, \(s=(b-a)/\max(a,b)\), where \(a\) is mean within-cluster distance and \(b\) is distance to the nearest other cluster. Values approach 1 for well-separated assignments. The selected `k=2` yields 0.406: useful structure, not definitive natural kinds. Profile tables are computed after fitting; survival is used only to interpret clusters, never to create them.

Pitfalls: sensitivity to scaling, spherical-cluster assumptions, arbitrary cluster IDs, and attaching stereotypes to descriptive groups.

## 4B. Anomaly detection

Isolation Forest repeatedly partitions random features. Rare points tend to be isolated in shorter paths. We fit only numeric behavioral/demographic values and choose contamination 0.05, so roughly 5% are flagged.

An anomaly score means **unusual under this feature representation**, not erroneous, fraudulent, or undesirable. The review list contains extreme fares and large families that may be valid. A production workflow would pair every flag with raw-record verification and a domain-specific action.

## 4C. Supervised learning

A stratified 75/25 split preserves the outcome ratio. Logistic regression supplies a linear log-odds baseline:

\[
P(y=1\mid x)=\sigma(\beta_0+\beta^T x),\quad \sigma(z)=\frac{1}{1+e^{-z}}.
\]

Random forest averages decorrelated decision trees to capture interactions. Accuracy is the correct-class fraction and matches Kaggle's competition metric. F1 is the harmonic mean of precision and recall. ROC-AUC is the probability that a randomly chosen positive is ranked above a randomly chosen negative.

Here logistic regression has ROC-AUC 0.842; random forest has 0.838 and the higher accuracy (0.798). “Best” therefore depends on the operational objective. A single split is appropriate for a quick curriculum, while a serious estimate would add repeated cross-validation, calibration, subgroup error analysis, and confidence intervals.

## 4D. Association rules

A transaction contains age band, fare tier, sex, class, embarkation, and outcome tokens. For rule \(X\rightarrow Y\):

- support: \(P(X\cap Y)\), how common the joint pattern is;
- confidence: \(P(Y\mid X)\), how often the consequent follows the antecedent;
- lift: \(P(Y\mid X)/P(Y)\), enrichment above baseline.

The bounded Apriori-style implementation enumerates itemsets of size 1–3, then retains rules with support at least 0.08, confidence at least 0.65, and lift above 1.05. A high-confidence rule can still be trivial when its consequent is common; lift supplies that missing baseline comparison. Multiple related rules are not independent discoveries.

## 4E. MinHash locality-sensitive hashing

Exact Jaccard similarity for sets \(A,B\) is \(|A\cap B|/|A\cup B|\). A full query scans every record, which is \(O(n)\). MinHash creates compact signatures whose coordinate agreement probability equals Jaccard similarity. LSH divides the signature into bands; matching an entire band places records into the same candidate bucket.

This project uses 48 hash permutations in 12 bands of 4 rows. The demonstration query examines 60 rather than 890 other records, then uses exact Jaccard similarity only within that set. This empirical candidate reduction demonstrates sub-linear *query work*, not a universal complexity guarantee: data distribution, banding, and similarity threshold affect recall and bucket size.

## 5. Evaluation

Evaluation asks whether technical evidence answers the original questions:

| Lens | Quality evidence | Valid conclusion | Invalid leap |
|---|---|---|---|
| Clustering | silhouette + profiles | two coarse recurring profiles | clusters are true social categories |
| Anomalies | scores + raw review | some profiles are rare numerically | flags are data errors |
| Supervised | held-out AUC/accuracy/F1 | features rank survival fairly well | features caused survival |
| Rules | support/confidence/lift | combinations co-occur above baseline | changing an item changes outcome |
| LSH | candidate reduction + exact rerank | similar profiles can be shortlisted | every true neighbor is guaranteed |

**Phase gate:** metrics are recomputed from held-out or method-appropriate evidence, results are stable enough for the stated scope, and caveats travel with the findings.

## 6. Deployment and monitoring

Deployment here is deliberately small: one command writes a versioned JSON-shaped evidence artifact; Flask serves a dashboard, `/api/results`, and `/health`; tests verify both analysis and delivery. In production, monitoring would track schema changes, missingness, category drift, outcome drift, subgroup errors, model calibration, anomaly review yield, LSH recall against a sampled exact search, runtime, and artifact freshness.

## Synthesis

The strongest lesson is methodological separation. Prediction, description, rare-case discovery, co-occurrence, and retrieval are different jobs. The Titanic features predict survival well enough for a classroom example and expose consistent structure around sex, class, and fare. Yet consistency across methods does not create causality: each algorithm transforms the same observational record through a different lens.

A defensible CRISP-DM conclusion therefore says: *the project meets its educational and reproducibility objectives; the results support historical descriptive and predictive claims within this sample; any policy, causal, or contemporary generalization requires new data and a different study design.*

## Review questions

1. What business decision would make recall more important than precision?
2. How would fitting median age before the split leak information?
3. Why is survival allowed in cluster profiles but not cluster inputs?
4. Which anomalies disappear if fare is log-transformed, and what does that teach us?
5. Construct a high-confidence rule with lift near 1; why is it weak?
6. How do more LSH bands change candidate recall and workload?
7. Which monitoring signal belongs to each CRISP-DM method?

