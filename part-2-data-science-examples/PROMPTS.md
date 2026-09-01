# Part 2 Prompt Log

This file preserves the meaningful Codex prompts used to reproduce each professor example. Add prompts and brief context under the corresponding project as the work is completed.

## Project 00: Dynamic Todo Workspace

In directory part-2-data-science-examples/00_dynamic_todo_workspace, build a modern end-to-end dynamic todo list application with industry-best UX and nice features. Make reasonable assumptions.

Keep the project self-contained, functional, and reasonably scoped. Add a concise README with run instructions. Test that it builds/runs successfully.

Do not modify Part 1. Do not commit or push.

## Project 01: NYC Taxi Trip Prediction

### Prompt

Work in `part-2-data-science-examples/01_nyc_taxi_trip_prediction`. Build an end-to-end data science project inspired by the Kaggle NYC taxi challenge, including data, training, deployment, the CRISP-DM framework, and an excellent frontend with an interactive map and trip estimation. Keep it reasonably scoped and runnable locally, using a manageable sample rather than excessive training or tuning. Add concise setup/run instructions and main results, and test the important functionality. Update the Part 2 index to mark Project 01 complete while preserving all other entries. Do not modify Part 1, commit, or push.

## Project 02: Nano LLM Transformer

### Prompt

Work in `part-2-data-science-examples/02_nano_llm_transformer`.

Build a simple LLM and chatbot using state-of-the-art primitives, but small enough to train and run on a normal laptop GPU. Follow the CRISP-DM framework and include a nice data science admin dashboard. You may research relevant papers and implement lightweight experimentation/hill climbing where practical. Include the details a data scientist and AI engineer would care about.

Keep training and experimentation reasonably small so the project can be reproduced locally without excessive compute or time. A small representative dataset and compact transformer are sufficient.

Add a concise README with setup, training, and run instructions. Test that training runs successfully, a saved model can be loaded, the chatbot produces generated text, and the dashboard/app starts. Update the Part 2 index to mark Project 02 complete and link its directory while preserving existing project entries. Record this prompt in `PROMPTS.md`. Do not modify Part 1, commit, or push.

### Prompt 2

NanoLlama is still giving mostly garbage output. Do a clean audit of the existing implementation, diagnose the training and generation pipeline, fix any bugs, and retrain with better synthetic or sample data if needed.
Keep the model small and practical to train locally. Do not redesign the project or expand scope unnecessarily.
After fixing it, test several prompts and show the resulting generations so we can verify that the output is noticeably more coherent.
Do not commit or push.

## Project 03: Customer Segmentation Clustering

### Prompt

Work in `part-2-data-science-examples/03_customer_segmentation_clustering`.

Build a clustering project using a popular Kaggle dataset. Follow the CRISP-DM framework and include a polished data science administration dashboard. Research relevant papers and implement a practical autoresearch hill-climbing loop, aligning dashboard metrics with the research. Include the details a data scientist and AI engineer would care about.

Keep the project reasonably scoped and runnable locally, using a manageable dataset or sample if needed. Add a concise README with setup/run instructions and key results. Test the clustering pipeline and dashboard. Update the Part 2 README to mark Project 03 complete and link its directory while preserving existing entries. Record this prompt in `PROMPTS.md`. Do not modify Part 1. Do not commit or push.

## Project 04: Associative Pattern Mining

### Prompt

Work in `part-2-data-science-examples/04_associative_pattern_mining`.

Build an associative pattern-mining project using a popular Kaggle dataset. Follow the CRISP-DM framework and include a polished data science administration dashboard. Research relevant papers and implement a practical autoresearch hill-climbing loop, aligning dashboard details with the research. Include the details a data scientist and AI engineer would care about.

Keep the project reasonably scoped and runnable locally, using a manageable dataset or sample if needed. Add a concise README with setup/run instructions and key results. Test the association/pattern-mining pipeline and dashboard. Update the Part 2 README to mark Project 04 complete and link its directory while preserving existing entries. Record this prompt in `PROMPTS.md`. Do not modify Part 1. Do not commit or push.

## Project 05: Data Science Skills Lab

### Prompt

Work in `part-2-data-science-examples/05_data_science_skills_lab`.
Install param087 GitHub `agent-ml-skills` and nimrodfisher `data-analytics-skills`, and demonstrate every skill on an appropriate popular Kaggle dataset. Include the CRISP-DM steps. Keep the implementation reasonably scoped and runnable locally, avoiding unnecessary expansion beyond demonstrating the requested skills. Add a concise README with setup/run instructions, test the application and skill demonstrations, update the Part 2 README to mark Project 05 complete and link its directory while preserving existing project entries, and record this prompt in `PROMPTS.md`. Do not modify Part 1. Do not commit or push.

## Project 06: Anomaly Detection

### Prompt

Work in `part-2-data-science-examples/06_anomaly_detection`.

Build an anomaly-detection project using a popular Kaggle dataset and popular methods. Follow the CRISP-DM framework and include a polished data science administration dashboard. Research relevant papers and implement a practical autoresearch hill-climbing loop, aligning dashboard details with the research. Include the details a data scientist and AI engineer would care about.

Keep the project reasonably scoped and runnable locally, using a manageable dataset or sample if needed. Add a concise README with setup/run instructions and key results. Test the anomaly-detection pipeline and dashboard. Update the Part 2 README to mark Project 06 complete and link its directory while preserving existing project entries. Record this prompt in `PROMPTS.md`. Do not modify Part 1. Do not commit or push.

## Project 07: AutoML AutoGluon

### Prompt

Work in `part-2-data-science-examples/07_automl_autogluon`.

Illustrate AutoML with AutoGluon on various data science tasks. Follow the CRISP-DM framework and include a polished data science administration dashboard. Research relevant papers and implement a practical autoresearch hill-climbing loop, aligning dashboard details with the research. Include the details a data scientist and AI engineer would care about.

Keep the project reasonably scoped and runnable locally. Use manageable datasets/samples and short AutoGluon training time limits so it does not require excessive compute. Add a concise README with setup/run instructions and key results. Test the AutoGluon training/inference workflow and dashboard. Update the Part 2 README to mark Project 07 complete and link its directory while preserving existing project entries. Record this prompt in `PROMPTS.md`. Do not modify Part 1. Do not commit or push.

## Project 08: Data Science Visual Mastery

### Prompt

Work in `part-2-data-science-examples/08_datascience_visual_mastery`.

Lets do another project - teach beginner data science students in an excellent way with deep intuition and rigorous math and visual intuition and live simulation on:

1. naive bayes
2. evaluation of model - confusion matrix, type 1 and type 2 errors, roc-auc, cost matrix, tradeoff between precision and recall
3. differential calculus, derivatives and how they connect to gradient descent
4. chain rule and how it connects to backpropagation

Include quizzes for each concept and interview prep questions. Also create a github.io ready page for this project.

Keep the implementation reasonably scoped and runnable locally. Make the visualizations and simulations interactive where practical.

Add a concise README with setup/run instructions.

Update `part-2-data-science-examples/README.md` to mark Project 08 complete and link to its directory. Preserve existing project entries.

Record this Project 08 prompt in `part-2-data-science-examples/PROMPTS.md`.

Do not modify Part 1. Do not commit or push.

When finished, briefly summarize what was implemented, how it was tested, and how to run it.

## Project 09: FlowForge DAG Engine

### Prompt

Work in `part-2-data-science-examples/09_flowforge_dag_engine`.
Install Matt Pocock skills and then demonstrate them with a complicated end-to-end full stack project.
Keep the project reasonably scoped and runnable locally. Build a clear full-stack demonstration without unnecessary infrastructure or scope expansion.
Add a concise README with setup/run instructions. Test the important frontend and backend functionality.
Update `part-2-data-science-examples/README.md` to mark Project 09 complete and link to its directory. Preserve existing project entries.
Record this Project 09 prompt in `part-2-data-science-examples/PROMPTS.md`.
Do not modify Part 1. Do not commit or push.
When finished, briefly summarize what was implemented, how it was tested, and how to run it.

## Projects 10-13

Add one section per project using this pattern:

```markdown
## Project NN: Project Name

### Prompt

<!-- Insert the meaningful reproduction prompt here. -->
```
