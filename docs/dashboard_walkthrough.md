# Interactive Dashboard Walkthrough

This guide provides a detailed walkthrough of the Power BI and Tableau dashboards developed for this project. These dashboards translate relational data into interactive visual stories to help HR stakeholders diagnose attrition trends.

---

## Dashboard Screenshots & Views
If you do not have Power BI Desktop or Tableau Reader installed, you can review the high-resolution screenshots saved in the repository under [dashboard/screenshots/](file:///home/shuhul/Employee-Attrition-and-Churn-Analysis-main/dashboard/screenshots).

### 1. Dashboard Overview
* **File Reference**: `dashboard-overview.jpg`
* **Purpose**: Serves as the primary landing page, highlighting macro-level statistics.
* **Key Components**:
  * **Top KPI Banner**: Displays four critical indicators:
    1. **Total Headcount**: Sum of all historical employees (Active + Attrition).
    2. **Attrition Count**: Total number of employees who have resigned.
    3. **Active Count**: Total number of current employees.
    4. **Attrition Rate**: Percentage of departures relative to total headcount.
  * **Department & Job Role Breakdown**: Treemaps or bar charts showcasing which business units account for the highest volume of departures.

### 2. Employee Demographics & Attrition Profile
* **File Reference**: `employee-overview.jpg`
* **Purpose**: Explores demographic attributes to identify vulnerability groups.
* **Key Components**:
  * **Gender and Marital Status Split**: Pie and donut charts highlighting gender-based distributions and marital status attrition levels (Single vs. Married).
  * **Travel Frequency analysis**: Vertical bar chart comparing travel frequency (Rarely, Frequently, Non-Travel) to turnover rates.
  * **Education Level Breakdown**: Analysis of attrition levels across educational fields (Medical, Life Sciences, Marketing, etc.).

### 3. Departmental & Role Deep Dive
* **File Reference**: `department-analysis.jpg`
* **Purpose**: Isolates specific teams and job titles that exhibit elevated attrition risk.
* **Key Components**:
  * **Role-Specific Turnover Rates**: Highlights high-risk job roles (e.g., Sales Executive, Lab Technician) through segmented horizontal bar charts.
  * **Average Tenure by Department**: Visual comparison of typical employee duration across business units.

### 4. Attrition Drivers & Compensation Analysis
* **File Reference**: `attrition-drivers.jpg`
* **Purpose**: Directly visualizes correlation matrices between compensation, workloads, and self-reported happiness.
* **Key Components**:
  * **Overtime Comparison**: Side-by-side bar chart showing attrition rates for employees working overtime vs. normal hours.
  * **Monthly Income Bracket Analysis**: Distribution chart grouping employees into salary ranges to show turnover at lower-tier pay levels.
  * **Work-Life Balance vs. Job Satisfaction Heatmap**: Displays how self-reported satisfaction intersects with workload ratings.

---

## How to Interact & Filter the Dashboards
To unlock the full potential of the dashboards, users can utilize the interactive filters:
1. **Macro Filters**: Located at the top or side pane of the dashboard. Users can filter the entire dataset by **Department**, **Gender**, and **Business Travel Frequency**.
2. **Cross-Filtering**: Clicking on any visual component (e.g., a specific department block in the Treemap) automatically filters all other charts on the page to focus on that segment.
3. **Tooltip Details**: Hovering over individual data points reveals underlying metrics, such as exact headcounts, averages, and percent-to-total ratios.
