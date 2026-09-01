# AI Coding Prompts

ChatGPT was used to help formulate and refine some of the prompts provided to Codex. Codex was then used as the primary coding agent to inspect the project, implement changes, run experiments, and assist with debugging.

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

## Prompt 4 - Creating a local demo app

Part 1 modeling is complete. Now add a very small local demo application for the existing What's Cooking cuisine classifier.

Goal:\
Create a simple Streamlit app that demonstrates the final model interactively for the assignment/video.

Important constraints:

- Keep this tiny and understandable.
- Do not redo EDA or cross-validation.
- Do not change the modeling conclusions or reported validation results.
- Do not modify raw files in data/.
- Do not modify the existing notebook unless absolutely necessary.
- Do not build an API, database, authentication system, React frontend, Docker setup, or deployment configuration.
- Do not commit or push anything.
- Reuse the same ingredient normalization and final Logistic Regression configuration used in the completed notebook.
- Avoid retraining the model every time Streamlit reruns.

Please first inspect:

- part-1-whats-cooking/notebooks/whats\_cooking\_analysis.ipynb
- part-1-whats-cooking/DATASET\_NOTES.md

Then implement the smallest clean solution.

Desired user experience:

1. Page title such as:\
   "What's Cooking? Cuisine Predictor"
2. Brief description:\
   Enter the ingredients of a recipe and the trained model will predict the most likely cuisine.
3. Ingredient input:
   - A multiline text area.
   - Allow ingredients separated by new lines or commas.
   - Include a reasonable example recipe as placeholder/example text.
4. A "Predict Cuisine" button.
5. After prediction, display:
   - Predicted cuisine prominently.
   - Top 3 predicted cuisines with probabilities if supported by the selected Logistic Regression model.
   - A short note explaining that this is a demonstration model trained on the Kaggle What's Cooking dataset and predictions are based only on ingredient phrases.
6. Handle simple input problems gracefully:
   - empty input;
   - duplicate ingredients;
   - extra whitespace;
   - capitalization;
   - ingredients unseen during training.

Implementation:

- Use Streamlit.
- Put the app at:\
  part-1-whats-cooking/app.py
- Put reusable model/preprocessing helpers under:\
  part-1-whats-cooking/src/
- Train the final app model only once using the full training dataset and the exact selected Logistic Regression configuration from the notebook.
- Save the fitted model as a local artifact under outputs/, for example:\
  part-1-whats-cooking/outputs/cuisine\_classifier.joblib
- The Streamlit app should load that saved model rather than retrain it on every launch.
- Make sure any custom preprocessing/vectorizer functions needed for joblib serialization live in an importable Python module rather than only inside app.py or a notebook.
- Keep ingredient phrases intact exactly as in the notebook's modeling approach.
- Do not use the recipe ID as a feature.

If a probability display is practical with the exact fitted Logistic Regression model, use predict\_proba() and show the top 3 classes. Do not add another model solely to obtain probabilities.

Also create or update a minimal requirements file for Part 1 if needed, containing only the packages necessary to run the notebook/app.

Add concise run instructions to:\
part-1-whats-cooking/README.md

For example, the final workflow should be approximately:

python src/train\_app\_model.py\
streamlit run app.py

If you can generate the saved model artifact directly during this task, do so, so that normal app use only requires:\
streamlit run app.py

Verification:

- Run the model preparation step.
- Launch or otherwise smoke-test the Streamlit app.
- Verify a sample recipe produces a prediction.
- Verify the saved model is loaded successfully.
- Verify empty input does not crash the app.

Do not redesign the entire README yet. Only add the minimum app-running section needed for now; we will do the full README as a separate task.

At the end report only:

- files created/modified;
- exact final model configuration reused;
- saved model artifact path and size;
- how to launch the app;
- one example input and prediction from the smoke test;
- whether the app passed the smoke test;
- any warnings or deviations.

Then stop.

## Prompt 5 - Interruption. Wrong environment used.

Stop. Do not install Python-3.8-specific Streamlit or Tornado versions.
My intended project environment is the Conda environment cmpe255-a1 using Python 3.11. VS Code and Jupyter are already configured to use it.
Before doing anything else, diagnose which Python YOUR shell is actually using.
Run and report:
python --version
where.exe python
python -c "import sys; print(sys.executable)"
python -m pip --version
If Conda is available, also run:
conda info --envs
Do not install or modify any packages yet.
If python is not the Python executable inside the cmpe255-a1 Conda environment, switch your commands to that environment rather than adapting the project to Python 3.8.
Prefer either:
conda run -n cmpe255-a1 python -m pip install streamlit
or use the exact Python 3.11 executable belonging to cmpe255-a1.
After identifying the correct Python 3.11 interpreter, install current Streamlit into THAT environment and verify:
python -c "import sys, streamlit; print(sys.executable); print(streamlit.version)"
Do not pin old Python-3.8-compatible Streamlit/Tornado versions.
Report the interpreter path before continuing with the app smoke test.

## Prompt 6 - Checking Notebook code against the new environment

Before doing any further retraining, review the existing Part 1 notebook for compatibility with the project's canonical environment:

- Conda environment: cmpe255-a1
- Python: 3.11.16
- scikit-learn: 1.9

Do NOT modify files, retrain models, regenerate outputs, or install packages yet.

Inspect:

part-1-whats-cooking/notebooks/whats\_cooking\_analysis.ipynb

and compare its code against the versions of the libraries currently installed in `cmpe255-a1`.

I want a focused compatibility audit.

For every code cell, identify any code that is:

- removed in the current library version;
- deprecated;
- behaviorally changed;
- likely to produce a warning;
- likely to fail under Python 3.11 / the currently installed package versions;
- or likely to produce materially different model behavior than in the original run.

Pay particular attention to:

- scikit-learn estimator constructor parameters;
- LogisticRegression multiclass behavior;
- solver behavior;
- class\_weight behavior;
- LinearSVC parameters;
- CountVectorizer behavior;
- StratifiedGroupKFold;
- cross\_val\_predict;
- classification\_report / confusion\_matrix;
- pandas APIs;
- NumPy APIs;
- matplotlib APIs.

For each issue, report:

1. Notebook cell number / section.
2. Existing code.
3. Whether it is:
   - ERROR in current environment,
   - DEPRECATED,
   - BEHAVIOR CHANGE,
   - or OK.
4. Recommended modern replacement.
5. Whether changing it would require retraining or could change the reported model scores.

Also report the current versions of:

- Python
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

Do not make any changes yet.

At the end give me a concise recommendation:

- notebook can remain as-is;
- notebook needs compatibility-only edits;
- or notebook should be rerun/retrained under Python 3.11.

Then stop.

## Prompt 7 - Fixing compatibility issues

The compatibility audit confirmed that the existing notebook was created against an older scikit-learn API and should be standardized on the canonical project environment.

Canonical environment:

- Conda environment: cmpe255-a1
- Python: 3.11.16
- scikit-learn: 1.9.0

Now update and rerun the existing notebook:

part-1-whats-cooking/notebooks/whats\_cooking\_analysis.ipynb

Do not recreate the notebook from scratch. Preserve its current structure, EDA, explanations, grouped-validation methodology, and generated artifacts unless a compatibility change requires an edit.

Important modeling decision:\
DATASET\_NOTES.md originally calls for multinomial Logistic Regression. Do not preserve the obsolete liblinear + multi\_class="ovr" implementation merely for backward compatibility.

Use a modern native multiclass Logistic Regression configuration supported by scikit-learn 1.9, such as:

LogisticRegression(\
C=1.0,\
solver="lbfgs",\
max\_iter=500,\
class\_weight="balanced",\
random\_state=RANDOM\_STATE\
)

Because this is a 20-class problem, let current scikit-learn use its native multinomial multiclass behavior. Do not specify the removed multi\_class parameter.

For LinearSVC, make the current behavior explicit:

LinearSVC(\
C=1.0,\
class\_weight="balanced",\
random\_state=RANDOM\_STATE,\
dual="auto"\
)

Keep:

- Majority-class baseline
- MultinomialNB(alpha=0.5)
- the same normalized full ingredient-phrase features
- the same recipe grouping
- the same 3-fold StratifiedGroupKFold
- the same random state
- the same leakage prevention

Do not expand this into a large hyperparameter search. The goal is environment standardization and reproducibility, not leaderboard optimization.

Update the influential-ingredient extraction so it works correctly with whichever final linear classifier is selected.

Then:

1. Run the entire notebook top-to-bottom using the cmpe255-a1 Python 3.11 environment.
2. Save the notebook with all execution counts and outputs.
3. Recompute all four model results.
4. Select the best model by grouped-validation accuracy.
5. Regenerate:
   - model\_comparison.csv
   - per-class report
   - normalized confusion matrix
   - common confusions
   - influential ingredients
   - submission.csv
6. Update all Markdown cells containing old hard-coded results so they match the new canonical run.
7. Make sure no absolute local filesystem paths are left in saved notebook output.
8. Do not modify raw data.
9. Do not commit or push anything.

After the notebook completes, rebuild the Streamlit app model artifact using the NEW validation-selected final model so the notebook and app use the same canonical model and environment.

Then smoke-test the existing app again.

At the end report:

- new results for all four models: accuracy and macro F1;
- old vs new Logistic Regression result;
- old vs new LinearSVC result;
- selected final model and exact parameters;
- whether all notebook cells executed successfully;
- whether submission.csv was regenerated;
- whether cuisine\_classifier.joblib was rebuilt from the same final model;
- one Streamlit smoke-test prediction;
- files modified;
- any remaining warnings or methodological deviations.

Then stop.

## Prompt 8 - Update DATASET_NOTES.md

Update:

part-1-whats-cooking/DATASET\_NOTES.md

to reflect the completed canonical experiment while preserving the distinction between the original modeling plan and what was actually implemented.

Do not rewrite the file from scratch and do not erase useful original dataset observations.

Make only documentation changes. Do not run models, modify the notebook, regenerate artifacts, modify the app, commit, or push anything.

Required changes:

1. Change the title from:\
   "Dataset Notes and Modeling Plan"\
   to something appropriate such as:\
   "Dataset Notes, Modeling Plan, and Final Implementation"
2. Preserve the dataset inspection, class-distribution, ingredient, duplicate/leakage, and data-quality findings unless they are factually outdated.
3. Rename "Recommended modeling strategy" to:\
   "Initial modeling strategy"

   Make clear that this was the plan established before modeling, not a claim about the final winning model.
4. Preserve the original intended methodology, but explicitly document the actual deviations rather than rewriting history:
   - planned 5-fold StratifiedGroupKFold;
   - final experiment used 3-fold StratifiedGroupKFold as a practical runtime compromise;
   - planned broader C/class-weight comparisons were intentionally not expanded into a large hyperparameter search.
5. Add a section:\
   "Canonical environment"

   Record:
   - Python 3.11.16
   - NumPy 2.4.6
   - pandas 3.0.5
   - scikit-learn 1.9.0
   - matplotlib 3.11.1
   - seaborn 0.13.2
   - joblib 1.6.0
6. Briefly explain that the original notebook had been produced against an older environment. After standardizing on Python 3.11 / scikit-learn 1.9, a compatibility audit found version-sensitive Logistic Regression and LinearSVC behavior. Therefore the entire notebook was rerun in the canonical environment and all final artifacts were regenerated.
7. Add a section:\
   "Final model comparison"

   Use these canonical 3-fold grouped-validation results:

   \| Model | Accuracy | Macro F1 |\
   \| Multinomial Naive Bayes | 0.7453 | 0.6527 |\
   \| Linear SVM | 0.7426 | 0.6552 |\
   \| Logistic Regression | 0.7405 | 0.6660 |\
   \| Majority-class baseline | 0.1971 | 0.0165 |

   Explain that:
   - accuracy remained the primary model-selection metric;
   - MultinomialNB(alpha=0.5) achieved the highest accuracy and was selected;
   - Logistic Regression achieved the highest macro F1, illustrating the tradeoff between overall accuracy and balanced per-class performance.
8. Document the final selected model:
   - binary CountVectorizer representation;
   - full normalized ingredient phrases;
   - lowercase / trim / within-recipe de-duplication;
   - MultinomialNB(alpha=0.5);
   - id excluded;
   - fitted on all training recipes only after model comparison.
9. State that the final model was used to regenerate:
   - outputs/submission.csv with 9,944 predictions;
   - outputs/cuisine\_classifier.joblib;
   - evaluation/interpretation artifacts;
   - and that the same fitted modeling approach is used by the Streamlit demo.
10. Replace the obsolete "Scope for the next step" section. It currently says that modeling/application work has not happened.

Replace it with a concise "Current project status" section stating that:

- EDA and model comparison are complete;
- the canonical notebook executes successfully top-to-bottom;
- the Kaggle-format submission is generated;
- a small Streamlit prediction app has been created and smoke-tested;

Do not add claims that are not supported by the existing project/results.

At the end, report what sections you changed. Then stop.

# Prompt 9 - Update README.md

Update:

part-1-whats-cooking/README.md

Documentation only. Do not retrain models, modify code/notebook/outputs, commit, or push.

Use the current project files as the source of truth, especially:

- DATASET\_NOTES.md
- PROMPTS.md
- notebooks/whats\_cooking\_analysis.ipynb
- app.py
- outputs/

Keep the README concise and suitable as the landing page for Part 1.

Include:

1. Project overview

- What's Cooking Kaggle dataset
- 39,774 train / 9,944 test / 20 cuisines
- goal: predict cuisine from ingredient lists
- AI-assisted data science workflow

2. Methodology

- normalized full ingredient phrases
- binary features
- 3-fold StratifiedGroupKFold with duplicate ingredient sets grouped
- models: Majority, MultinomialNB, Logistic Regression, LinearSVC

3. Final results

| ModelAccuracyMacro F1   |        |        |
| ----------------------- | ------ | ------ |
| Multinomial Naive Bayes | 0.7453 | 0.6527 |
| Linear SVM              | 0.7426 | 0.6552 |
| Logistic Regression     | 0.7405 | 0.6660 |
| Majority baseline       | 0.1971 | 0.0165 |

State that MultinomialNB(alpha=0.5) was selected by accuracy, while Logistic Regression had the best macro F1.

Embed 2-3 useful existing plots from outputs/.

4. AI workflow\
   Briefly state:

- ChatGPT helped with planning/review/prompt refinement
- Codex handled agentic implementation/debugging
- I reviewed and verified the results
- standardizing on Python 3.11 / scikit-learn 1.9 changed the final model ranking, showing why AI output must be checked

Link to PROMPTS.md.

5. Streamlit demo\
   Briefly describe app.py and include:

conda activate cmpe255-a1\
cd part-1-whats-cooking\
streamlit run app.py

6. Main artifacts\
   Link to:

- notebook
- DATASET\_NOTES.md
- PROMPTS.md
- app.py
- submission.csv
- model\_comparison.csv
- article.md



Use relative GitHub links only.\
Do not expose local paths.\
Do not duplicate large parts of DATASET\_NOTES.md or the notebook.\
Keep it readable in about 2-3 minutes.

Then stop.

## Prompt 10 - Write a Medium article

Write:

part-1-whats-cooking/article.md

The article should not be only a technical write-up of the What's Cooking classifier. The main theme should be the end-to-end experience of using AI agents for a data science assignment: planning, prompting, implementation, debugging, verification, reproducibility, and final modeling results.

Use the existing project files as the source of truth, especially:

* README.md
* DATASET_NOTES.md
* PROMPTS.md
* notebooks/whats_cooking_analysis.ipynb
* app.py
* outputs/

Do not retrain models or modify project code.

Write in a natural first-person student voice suitable for a Medium-style technical article. Avoid sounding promotional or overly polished.

Suggested structure:

# Building a Data Science Project with AI Agents: What Worked, What Broke, and What I Learned

## Introduction

Briefly explain the CMPE 255 assignment and why I chose the Kaggle What's Cooking dataset.

Frame the article around a question such as:
What actually happens when an AI coding agent is used to build an end-to-end data science project?

## My AI-Assisted Workflow

Describe the real workflow:

* ChatGPT helped with planning, reviewing methodology, and refining prompts.
* Codex in VS Code acted as the main coding agent.
* Codex inspected the dataset, created the notebook, implemented modeling, generated artifacts, and later built the Streamlit demo.
* I reviewed the generated work instead of treating the agent as automatically correct.

Link this discussion to PROMPTS.md where useful.

## Understanding the Dataset Before Modeling

Summarize only the most important findings:

* 39,774 training recipes
* 9,944 test recipes
* 20 cuisines
* class imbalance
* ingredient phrases as the main input
* duplicate ingredient sets and why grouped validation was important

Keep this section concise.

## Modeling Approach

Explain the final methodology:

* normalized full ingredient phrases
* binary CountVectorizer features
* 3-fold StratifiedGroupKFold
* Majority baseline
* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM
* accuracy as the primary metric
* macro F1 as a secondary metric

## The Most Important Part: Verifying the Agent

This should be one of the central sections.

Describe what actually happened:

* the first notebook was created against an older Python/scikit-learn environment;
* after I switched the project to a Conda Python 3.11 environment, compatibility issues appeared;
* Logistic Regression used an API that was removed/changed in newer scikit-learn;
* LinearSVC behavior was also version-sensitive;
* there were also notebook execution/path/kernel issues;
* I manually ran and inspected the notebook instead of assuming the agent's "successful execution" claim was enough;
* the project was standardized on Python 3.11.16 / scikit-learn 1.9 and rerun completely.

Explain why this matters:
AI can generate plausible code quickly, but reproducibility, dependency versions, execution environment, and methodological correctness still require human verification.

## Final Canonical Results

Use the final results:

| Model                   | Accuracy | Macro F1 |
| ----------------------- | -------: | -------: |
| Multinomial Naive Bayes |   0.7453 |   0.6527 |
| Linear SVM              |   0.7426 |   0.6552 |
| Logistic Regression     |   0.7405 |   0.6660 |
| Majority baseline       |   0.1971 |   0.0165 |

Explain that:

* MultinomialNB(alpha=0.5) was selected because it had the highest validation accuracy;
* Logistic Regression had the highest macro F1;
* the top models were close;
* the final ranking was different from the initial older-environment run.

Do not present obsolete pre-standardization scores as final results.

## From Notebook to Small App

Briefly describe the Streamlit demo:

* user enters recipe ingredients;
* app predicts cuisine;
* displays top predicted cuisines/model probabilities;
* app uses the same final selected modeling approach.

Keep this section short. The article is primarily about the AI-assisted data science process, not Streamlit development.

## What I Learned

Write this in first person and make it substantive.

Cover ideas such as:

* good prompts help constrain an agent, but prompt quality alone does not guarantee correctness;
* breaking work into bounded tasks made the workflow easier to control;
* AI was very effective at scaffolding, repetitive coding, and debugging;
* I still needed to understand validation, leakage, metrics, environment compatibility, and model selection;
* manually verifying execution exposed issues that the agent initially missed;
* reproducibility is part of data science, not merely a setup detail;
* AI agents are most useful when treated as capable collaborators whose work must be reviewed.

## Conclusion

Conclude with a balanced view:
AI significantly accelerated the project, but the most important lesson was that using an agent does not remove the need for understanding, verification, and methodological judgment.

Additional requirements:

* Keep the article roughly 1,500-2,000 words.
* Use a natural first-person voice.
* Do not invent events, results, or decisions.
* Do not make it sound like Codex independently completed the assignment without human involvement.
* Avoid generic AI hype.
* Include a few useful existing figures from outputs/ using relative Markdown image links where they strengthen the story.
* Prefer 2-3 figures rather than turning the article into a gallery.
* Use only relative repository links.
* Do not expose local Windows paths.
* Do not modify any other project files.
* Do not commit or push.

At the end report:

* article title;
* approximate word count;
* which figures were included;
* any facts you could not verify from the project files.

Then stop.
