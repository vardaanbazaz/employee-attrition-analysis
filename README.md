# Employee Attrition Analysis & Predictive Modeling

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![SQL: MySQL 8.0](https://img.shields.io/badge/SQL-MySQL%208.0-orange.svg)](#)
[![Machine Learning: Scikit--Learn / XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](#)
[![BI Tools: PowerBI & Tableau](https://img.shields.io/badge/BI%20Tools-Power%20BI%20%7C%20Tableau-764ABC.svg)](#)

> **Series-A Production-Grade Data Science, Exploratory Data Analysis (EDA) & Predictive Modeling Solution** for enterprise human capital management. Combines a normalized MySQL database schema, high-performance diagnostic SQL analytics, machine learning classifiers (Random Forest & XGBoost with SMOTE oversampling), and interactive executive dashboards (Power BI & Tableau) to diagnose, quantify, and predict employee flight risk.

---

## 💡 Executive Summary

Unplanned employee turnover represents one of the largest hidden operational and financial drains on modern enterprise organizations. Research demonstrates that replacing a departed employee costs between **1.5x to 2.0x their annual base salary** when accounting for recruitment advertising, interviewing cycles, onboarding overhead, temporary productivity gaps, and institutional knowledge loss.

This **Employee Attrition Analysis & Predictive Analytics Platform** transforms fragmented, multi-source HR telemetry into an end-to-end data science and decision-support pipeline:

* **Quantifying Attrition Triggers**: Isolate high-vulnerability cohorts experiencing elevated churn due to workload intensity, compensation stagnation, or lack of promotion velocity.
* **Predictive Churn Risk Scoring**: Leverage supervised machine learning algorithms trained on balanced employee profiles (via SMOTE) to predict flight risk before resignation notices occur.
* **Data-Driven Executive Action**: Supply HR leadership and department executives with empirical recommendations to optimize pay structures, enforce overtime thresholds, and redesign onboarding milestones.

---

## 🛠️ Step 1: Data Engineering & Relational Database Schema

The foundational layer of the analytical pipeline ingests raw tabular datasets (`HR_1.xlsx` demographics/satisfaction metrics and `HR_2.xlsx` compensation/tenure stats) into a normalized **MySQL 8.0** relational database.

![Technical Architecture](assets/architecture.png)

### Database Pipeline & Relational DDL
* **Schema Design**: Implemented in [`sql/schema.sql`](sql/schema.sql), splitting 50,000 synthetic employee records into two normalized entities:
  * `hr_1`: Captures demographics (`Age`, `Gender`, `MaritalStatus`, `Department`, `JobRole`) and 1–4 ordinal satisfaction scores (`EnvironmentSatisfaction`, `JobSatisfaction`, `JobInvolvement`).
  * `hr_2`: Captures compensation (`MonthlyIncome`, `PercentSalaryHike`), workload (`OverTime`), and tenure metrics (`YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`).
* **Referential Integrity**: Linked via `FOREIGN KEY (EmployeeID) REFERENCES hr_1(EmployeeNumber) ON DELETE CASCADE` to guarantee data consistency across joins.

---

## 🔍 Step 2: Diagnostic SQL Analytics

The core analytical queries stored in [`sql/employee_attrition_analysis.sql`](sql/employee_attrition_analysis.sql) aggregate metrics across demographic, financial, and operational dimensions.

### Query A: Attrition Rate by Monthly Income Bracket
*Categorizes employees into logical salary tiers to identify income-driven turnover thresholds:*

```sql
SELECT 
    CASE 
        WHEN t2.monthlyincome < 3000 THEN 'Under $3k'
        WHEN t2.monthlyincome BETWEEN 3000 AND 6000 THEN '$3k - $6k'
        WHEN t2.monthlyincome BETWEEN 6001 AND 10000 THEN '$6k - $10k'
        WHEN t2.monthlyincome BETWEEN 10001 AND 15000 THEN '$10k - $15k'
        ELSE 'Over $15k'
    END AS income_bracket,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY income_bracket
ORDER BY MIN(t2.monthlyincome) ASC;
```

### Query B: Burnout Risk Cohort Analysis
*Evaluates the compounding risk of excessive workload (Overtime) combined with low compensation raises (<15%):*

```sql
SELECT 
    t2.overtime,
    CASE 
        WHEN t2.percentsalaryhike < 15 THEN 'Low Hike (<15%)' 
        ELSE 'Normal/High Hike (15%+)' 
    END AS hike_status,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t2.overtime, hike_status
ORDER BY attrition_rate_pct DESC;
```

---

## 🤖 Step 3: Predictive Machine Learning Pipeline

Beyond descriptive SQL diagnostics, this platform incorporates an end-to-end Machine Learning pipeline implemented in [`notebooks/employee_churn_prediction.ipynb`](notebooks/employee_churn_prediction.ipynb).

### Pipeline Workflow
1. **Exploratory Data Analysis (EDA)**: Analyzes univariate distribution features and multivariate correlations across employee attributes.
2. **Preprocessing & Encoding**: Performs one-hot encoding on categorical variables (`BusinessTravel`, `Department`, `JobRole`, `MaritalStatus`) and standardizes continuous numerical attributes.
3. **Addressing Class Imbalance (SMOTE)**: Employee attrition data exhibits severe class imbalance (majority of employees stay). We apply **Synthetic Minority Over-sampling Technique (SMOTE)** to synthesize minority class samples and prevent model bias toward majority classes.
4. **Model Training & Benchmarking**: Evaluates multiple supervised classifiers:
   * **Logistic Regression**: Serves as a baseline interpretable linear classifier.
   * **Random Forest Classifier**: Captures non-linear feature interactions and tree-based feature importance.
   * **XGBoost Classifier**: Delivers peak predictive accuracy, high ROC-AUC, and fine-tuned precision/recall tradeoffs for candidate flight risk.
5. **Churn Probability Generation**: Exports binary classification models capable of outputting probability scores (`0.0 - 1.0`) for proactive HR retention interventions.

---

## 📊 Step 4: Business Intelligence Dashboards

The visual analytics layer features a 4-view dashboard gallery served via **Power BI** (`dashboard/powerbi/employee_attrition_dashboard.pbix`) and **Tableau** (`dashboard/tableau/employee_attrition_dashboard.twbx`).

### 1. Executive Overview Dashboard
![Executive Overview Dashboard](dashboard/screenshots/dashboard-overview.jpg)
* **Description**: Primary landing view displaying macro KPI indicators (Total Headcount, Attrition Count, Active Count, Attrition Rate %) alongside high-level departmental exit distributions.

---

### 2. Departmental & Role Diagnostics
![Departmental Diagnostics](dashboard/screenshots/department-analysis.jpg)
* **Description**: Granular breakdown isolating high-risk job titles (*Sales Executives*, *Lab Technicians*, *Research Scientists*) and comparing average organizational tenure across departments.

---

### 3. Attrition Drivers & Compensation Heatmaps
![Attrition Drivers](dashboard/screenshots/attrition-drivers.jpg)
* **Description**: Visualizes multivariate drivers of attrition, illustrating the impact of overtime hours, salary bracket distribution, and work-life balance scores.

---

### 4. Employee Demographics & Workforce Profile
![Employee Overview](dashboard/screenshots/employee-overview.jpg)
* **Description**: Explores demographic attributes including age distributions, business travel frequency, education fields, and marital status correlations.

---

## 📈 Step 5: Key Business Insights & Strategic Recommendations

Extracted directly from comprehensive analytical reports ([`docs/insights.md`](docs/insights.md)):

### Strategic Findings
1. **Overtime Burnout**: Employees working overtime depart at **more than double the rate** of non-overtime staff. Overtime employees receiving `<15%` salary hikes experience over **60% attrition**.
2. **Compensation Ceiling**: Attrition is heavily concentrated among staff earning **under $3,000 per month**. Salary raises below 15% fail to function as effective retention drivers.
3. **Role Volatility**: Three specific roles (*Sales Executives*, *Laboratory Technicians*, and *Research Scientists*) account for **over 50% of all company exits**.
4. **Early-Tenure Vulnerability**: Employees in their first **0–24 months** show peak attrition, signaling critical gaps in initial onboarding and early career growth alignment.

### Strategic Recommendations
* 🛡️ **Enforce Workload Caps**: Mandate departmental overtime limits and require automated HR check-ins whenever an employee exceeds 10 overtime hours per week.
* 💵 **Market-Rate Pay Adjustments**: Conduct compensation benchmarks for low-earning roles and ensure high performers receive a minimum 15% annual salary hike.
* 🎯 **Structured Tenure Milestones**: Implement formal 12-month and 24-month career trajectory planning to give mid-tenure employees visible growth pathways.

---

## 📁 Repository Architecture

```
employee-attrition-analysis/
│
├── README.md                          # Single source of truth documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Version control exclusions
│
├── assets/
│   └── architecture.png               # High-resolution ETL & machine learning architecture diagram
│
├── sql/
│   ├── schema.sql                     # DDL scripts for normalized MySQL schema
│   └── employee_attrition_analysis.sql # Diagnostic SQL analytical suite
│
├── data/
│   ├── raw/
│   │   ├── HR_1.xlsx                 # Primary demographics & satisfaction metrics
│   │   └── HR_2.xlsx                 # Primary compensation & tenure metrics
│   └── optional/
│       └── churn_prediction_dataset/ # Supplementary dataset for predictive ML models
│
├── dashboard/
│   ├── powerbi/
│   │   └── employee_attrition_dashboard.pbix   # Power BI workbook
│   ├── tableau/
│   │   └── employee_attrition_dashboard.twbx   # Tableau packaged workbook
│   └── screenshots/
│       ├── dashboard-overview.jpg     # Executive KPI & macro overview view
│       ├── department-analysis.jpg    # Department & role deep-dive view
│       ├── attrition-drivers.jpg      # Overtime & compensation heatmap view
│       └── employee-overview.jpg      # Demographics & workforce profile view
│
├── notebooks/
│   └── employee_churn_prediction.ipynb # EDA, SMOTE oversampling, & predictive ML pipeline
│
└── docs/
    ├── business_problem.md            # Financial cost & business context documentation
    ├── methodology.md                 # Technical ETL, SQL design, & ML methodology
    ├── insights.md                    # Detailed strategic findings & recommendations
    └── dashboard_walkthrough.md       # Interactive dashboard usage guide
```

---

## 🚀 Quick Start Guide

### 1. Database Setup
```bash
# Clone the repository
git clone https://github.com/your-username/employee-attrition-analysis.git
cd employee-attrition-analysis

# Import relational schema and diagnostic queries into MySQL
mysql -u root -p < sql/schema.sql
mysql -u root -p < sql/employee_attrition_analysis.sql
```

### 2. Machine Learning Execution
```bash
# Launch Jupyter Notebook to train models and run EDA
jupyter notebook notebooks/employee_churn_prediction.ipynb
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
