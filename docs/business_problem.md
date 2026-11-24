# Business Problem: Employee Attrition

## Question
Which recorded attributes of an employee (overtime, tenure, income, satisfaction scores and so on) are associated with leaving a company, and how well can a standard classifier separate leavers from stayers on those attributes?

## Why it matters
Turnover has direct costs (recruiting, training), indirect costs (extra load on the remaining team) and knowledge loss when people leave. An HR team that understands which patterns go with attrition can decide where to look more closely.

## The data
IBM's HR Analytics Employee Attrition dataset: 1,470 rows, 35 columns, **fictional**, created by IBM data scientists. It is not a record of real employees, so every finding describes this dataset only. About 16% of rows are leavers.

## What the project does
* **Descriptive**: attrition rates by department and income, overtime split, numeric correlations (`scripts/generate_visuals.py`, figures in `assets/eda_stats.json`).
* **Predictive**: compares a dummy baseline, Logistic Regression, Random Forest and XGBoost by 5-fold CV PR-AUC on the training split, then reports test-set results (`assets/metrics.json`).
* **Explanatory**: SHAP values show which features the selected model relies on.

## What the model can and cannot support
* It can show which features the model uses and in which direction, on this dataset.
* It cannot identify employees who will resign. At the 0.5 threshold the selected model finds a minority of leavers (test recall in `assets/metrics.json`), and the two best models are tied within CV noise.
* SHAP shows associations the model learned, not causes. It does not show that changing a feature would change whether someone leaves.
* It is a case study on fictional data, not a validated tool for real HR decisions.
