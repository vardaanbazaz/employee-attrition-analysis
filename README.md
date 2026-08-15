# Enterprise Employee Attrition: ML Prediction & Diagnostic Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-4B0082?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An enterprise-grade, end-to-end Machine Learning and Diagnostic Analytics platform designed to proactively identify employee flight risk, quantify compensation & workload friction points, and provide model transparency using **SHAP (SHapley Additive exPlanations)**.

---

## 📌 Executive Summary

Employee turnover costs enterprises millions annually in lost productivity, onboarding overhead, and institutional knowledge drain. Traditional HR approaches rely on retrospective exit surveys after an employee has already resigned.

This platform transitions workforce management from reactive tracking to **predictive intelligence**. Built as a modular Python ML engine, it ingests multi-source HR telemetry (demographics, compensation, performance, tenure), resolves class imbalance using **SMOTE**, trains gradient-boosted decision tree models (**XGBoost** & **Random Forest**), and decodes model decisions down to individual feature attributions via **SHAP**.

### Key Business Metrics & Machine Learning Pipeline Wins
* **Dataset Scale**: Processed and merged **50,000 employee records** across 36 tabular features.
* **Class Imbalance Resolution**: Applied SMOTE oversampling to achieve a balanced **35,148 training set** (`50% / 50%` target split).
* **Predictive Performance**: Evaluated classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) across Logistic Regression baselines, Random Forest, and XGBoost classifiers.
* **Explainability Coverage**: Integrated 100% SHAP feature attribution mapping, translating black-box ML outputs into actionable retention levers for HR leadership.

---

## 🏗️ Pipeline Architecture & Modular System Design (`src/`)

The core repository architecture decouples the machine learning workflow into a modular, production-ready Python pipeline located in [`src/`](src/):

```
src/
├── data_ingestion.py      # Multi-source data ingestion, schema validation, & relational joins
├── preprocessing.py       # Feature encoding, StandardScaler, & SMOTE class balancing
├── train.py               # Classifier training (XGBoost, Random Forest, Logistic Regression) & pickle serialization
└── evaluate.py            # Diagnostic evaluation reporting & SHAP explainability plot generation
```

### 1. Data Ingestion ([`src/data_ingestion.py`](src/data_ingestion.py))
* **Primary Key Relational Join**: Automates the joining of tabular demographics (`data/raw/HR_1.xlsx`) and compensation/performance metrics (`data/raw/HR_2.xlsx`) on `EmployeeNumber` ↔ `Employee ID`.
* **Data Hygiene**: Cleans trailing unformatted Excel columns and validates column schemas prior to downstream processing.

### 2. Preprocessing & Imbalance Handling ([`src/preprocessing.py`](src/preprocessing.py))
* **Feature Encoding**: Converts categorical string columns (`Department`, `JobRole`, `OverTime`, `BusinessTravel`, etc.) into model-ready integers using `LabelEncoder`.
* **Feature Standardization**: Applies `StandardScaler` to ensure zero mean and unit variance across continuous numerical inputs (`MonthlyIncome`, `Age`, `YearsAtCompany`).
* **SMOTE Oversampling**: Leverages `imbalanced-learn`'s Synthetic Minority Over-sampling Technique (SMOTE) on the training set to prevent model bias towards majority retained classes.

### 3. Model Training & Serialization ([`src/train.py`](src/train.py))
* **Classifier Suites**: Supports baseline Logistic Regression, Random Forest, and XGBoost hyperparameterized classifiers.
* **Artifact Serialization**: Automatically exports fitted model binaries (`.pkl`) to `models/` for downstream inference services.

### 4. Evaluation & Diagnostic Diagnostics ([`src/evaluate.py`](src/evaluate.py))
* **Metric Reporting**: Computes Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices on unseen test datasets (`15,000 records`).
* **SHAP Integration**: Automatically triggers SHAP value calculation and summary plot generation.

---

## 📊 Exploratory Data Analysis (EDA) & Dark-Mode Analytics

Custom dark-mode visual telemetry generated programmatically using `pandas`, `matplotlib`, and `seaborn` ([`scripts/generate_visuals.py`](scripts/generate_visuals.py)):

### 1. Departmental Attrition Diagnostics
![Departmental Attrition Rate](assets/attrition_by_department.png)
* **Insight**: Quantifies percentage turnover and exit counts across departments. Highlights elevated attrition in high-strain units like Research & Development and Software Engineering.

---

### 2. Compensation Density & Flight Risk Analysis
![Compensation Density vs Attrition Risk](assets/compensation_vs_attrition.png)
* **Insight**: Kernel Density Estimation (KDE) plot comparing income distributions of retained vs. departed staff. Attrition is heavily concentrated among staff earning under **$3,000/month**.

---

### 3. Numerical Telemetry Correlation Matrix
![Numerical Telemetry Correlation Matrix](assets/correlation_heatmap.png)
* **Insight**: Multi-variable correlation heatmap isolating strong interactions between age, total working years, monthly income, salary hike percentage, and flight risk probability.

---

## 🧠 Machine Learning Explainability (SHAP)

To transition from "black-box" predictions to actionable business intelligence, the pipeline leverages **SHAP (SHapley Additive exPlanations)** based on cooperative game theory.

![SHAP Feature Importance Summary Plot](assets/shap_summary.png)

### Demystifying the "Black Box": Flight Risk Attribution
The SHAP Summary Plot maps both the **magnitude** and **direction** of feature influences on the XGBoost classifier's flight risk predictions:

1. **OverTime (Primary Risk Driver)**:
   - High overtime frequency (red dots extending to the right) exerts the strongest positive push on SHAP values, significantly elevating predicted flight risk.
2. **Monthly Income & Daily Rates**:
   - Low monthly income (`MonthlyIncome`) and low daily rates (`DailyRate`) push SHAP scores into the positive churn domain, proving that under-compensated employees face higher flight risk.
3. **Tenure & Manager Stability**:
   - Brief tenure (`YearsAtCompany`) and short relationships with current managers (`YearsWithCurrManager`) combined with long commutes (`DistanceFromHome`) strongly amplify churn probability.

By evaluating these individual Shapley attributions, HR leaders can understand *why* an employee is flagged as high-risk and implement targeted retention interventions (e.g., overtime caps or salary benchmarking) before an exit occurs.

---

## 💻 Setup & Installation Guide

### Prerequisites
* Python 3.10+
* Virtual Environment (`venv` or `conda`)

### 1. Clone & Set Up Environment
```bash
# Clone repository
git clone https://github.com/your-username/employee-attrition-analysis.git
cd employee-attrition-analysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Modular ML Pipeline
```bash
# Run model training workflow
python src/train.py

# Run full evaluation, metric diagnostics, and SHAP plot generation
python src/evaluate.py
```

### 3. Generate Analytics Visuals
```bash
# Programmatically regenerate dark-mode EDA plots in assets/
python scripts/generate_visuals.py
```

---

## 📁 Repository Structure

```
employee-attrition-analysis/
│
├── README.md                          # Master project documentation
├── requirements.txt                   # Production Python dependencies manifest
├── pyrightconfig.json                 # Language server environment configuration
├── LICENSE                            # MIT License
├── .gitignore                         # Version control exclusions
│
├── src/                               # Modular Machine Learning Pipeline
│   ├── data_ingestion.py              # Ingests & relationally merges raw Excel files
│   ├── preprocessing.py               # Label encoding, scaling, & SMOTE oversampling
│   ├── train.py                       # Fits XGBoost, Random Forest, & Logistic Regression
│   └── evaluate.py                    # Metric diagnostics & SHAP explainability plot generator
│
├── scripts/
│   └── generate_visuals.py            # Programmatic dark-mode analytics plot generator
│
├── assets/
│   ├── shap_summary.png               # SHAP feature importance & flight risk summary plot
│   ├── attrition_by_department.png    # Dark-mode departmental attrition rate bar chart
│   ├── compensation_vs_attrition.png  # Dark-mode compensation KDE density plot
│   └── correlation_heatmap.png        # Dark-mode numerical feature correlation matrix
│
├── sql/
│   ├── schema.sql                     # DDL scripts for normalized MySQL database
│   └── employee_attrition_analysis.sql # Diagnostic SQL query suite
│
├── data/
│   └── raw/
│       ├── HR_1.xlsx                  # Demographics & satisfaction dataset
│       └── HR_2.xlsx                  # Compensation & tenure dataset
│
├── models/                            # Serialized model binaries (.pkl)
│   └── xgboost_model.pkl
│
└── notebooks/
    └── employee_churn_prediction.ipynb # Interactive EDA & prototyping notebook
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
