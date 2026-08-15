# Technical Methodology & Data Pipeline

This section documents the end-to-end technical approach used to process, store, analyze, and visualize the employee attrition and churn datasets.

---

## 1. Data Source & Architecture
The project utilizes two primary data domains:
1. **Employee Attrition Domain (Core)**: 
   * **Source Files**: `HR_1.xlsx` and `HR_2.xlsx` containing demographics, performance metrics, and salary figures.
   * **Volume**: ~50,000 synthetic employee records.
   * **Structure**: Normalized into a relational database schema.
2. **Candidate Churn Domain (Optional/Predictive)**:
   * **Source Files**: `churn.csv` and `churn_test.csv`.
   * **Volume**: ~19,000 candidate records.
   * **Structure**: Used for building and testing predictive machine learning models in Python.

The architecture is outlined below:
```
Raw Data Files (Excel/CSV) ──> SQL Database Normalization (MySQL) ──> Business BI Dashboard (Power BI / Tableau)
                                                                 └──> Predictive Modeling (Jupyter Notebook)
```

---

## 2. ETL (Extract, Transform, Load) & Database Ingestion
To ensure clean data representation and optimal query performance, the spreadsheet data was ingested and normalized into a MySQL database schema using the following steps:
* **Table Splitting & Normalization**: The data was split into two entities to maintain database cleanliness:
  * `hr_1` (Demographics and Role): Stores immutable or slow-moving demographic attributes (Age, Gender, Marital Status) and job details (Department, Role, Satisfaction scores).
  * `hr_2` (Performance and Compensation): Stores volatile metrics, financial information (Monthly Income, Hike percentages), and tenure details.
  * **Primary Key Mapping**: `hr_1.EmployeeNumber` maps 1-to-1 to `hr_2.EmployeeID` via a Foreign Key relationship with cascading deletes.
* **Standardization**:
  * Cleaned inconsistent nulls and empty cells.
  * Mapped string-based levels to categorical values where appropriate.
  * Standardized column types (e.g. integer fields for scores 1-4, exact datatypes for monetary counts).

---

## 3. SQL Diagnostic Analytics
Using a structured suite of queries, we analyzed the joined tables to segment populations. This was performed across:
* **Aggregation & Ratios**: Calculating total percentages instead of absolute counts to normalize comparisons (e.g., Attrition Rate % rather than simple count).
* **Conditional Binning**: Using `CASE WHEN` constructs in SQL to bucket continuous values (e.g., Monthly Income into income brackets, Tenure into joiner classifications) to discover non-linear trends.
* **Sub-Group Analysis**: Combining multiple stress indicators (e.g., Overtime work AND Low Salary Hike) to isolate high-burnout employee profiles.

---

## 4. Dashboard Design Principles
The Tableau (`.twbx`) and Power BI (`.pbix`) dashboards were built following modern UI/UX design standards:
* **KPI Card Placement**: Summary metrics (Total Headcount, Attrition Count, Active Employees, Attrition Rate) are placed prominently at the top-left of the view.
* **Color Hierarchy**: A curated, high-contrast palette is used. Neutral shades are employed for standard metrics, while a dedicated alert color represents Attrition/Departure indicators to immediately draw eye attention.
* **Interactivity**: Page filters (Department, Gender, Travel Frequency) and cross-filtering between chart components allow users to dynamically segment and drill down into the dashboard.

---

## 5. Machine Learning Candidate Churn Modeling (Optional)
In parallel to descriptive BI, predictive analytics was performed on the candidate churn dataset in `notebooks/employee_churn_prediction.ipynb`:
* **Imbalance Handling**: The dataset suffered from class imbalance. SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) was implemented to balance the target class.
* **Model Evaluation**: Multiple classifiers (Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting) were trained and evaluated.
* **Selection Metric**: Recall was prioritized to minimize False Negatives (missing candidates likely to leave).
