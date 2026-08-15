-- =========================================================================
-- EMPLOYEE ATTRITION & PERFORMANCE ANALYTICS QUERY SUITE
-- Goal: Analyze key organizational drivers of employee attrition.
-- Database Compatibility: MySQL (ANSI-SQL compliant)
-- Schema: Normalized hr_1 (Demographics) and hr_2 (Performance & Finance)
-- =========================================================================

USE hrdataanalytics;

-- =========================================================================
-- SECTION 1: ATTRITION RATE METRICS
-- Summarize overall attrition and initial demographic indicators.
-- =========================================================================

-- Query 1.1: Overall Headcount, Attrition counts, and Attrition Rate
-- Provides a high-level baseline of how many employees left the company.
SELECT 
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    SUM(CASE WHEN attrition = 'No' THEN 1 ELSE 0 END) AS active_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS overall_attrition_rate_pct
FROM hr_1;


-- Query 1.2: Attrition Rate by Business Travel Frequency
-- Tests the hypothesis that heavy travel requirements lead to higher attrition.
SELECT 
    businesstravel,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY businesstravel
ORDER BY attrition_rate_pct DESC;


-- Query 1.3: Attrition Rate by Gender
-- Checks if there is a significant discrepancy in attrition between genders.
SELECT 
    gender,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
WHERE gender IS NOT NULL AND gender <> 'NaN'
GROUP BY gender
ORDER BY attrition_rate_pct DESC;


-- =========================================================================
-- SECTION 2: DEPARTMENT & ROLE ANALYSIS
-- Drill down into organizational units to find structural attrition pockets.
-- =========================================================================

-- Query 2.1: Attrition Rate by Department
-- Identifies which macro-organizational units suffer from the highest turnover.
SELECT 
    department,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY department
ORDER BY attrition_rate_pct DESC;


-- Query 2.2: Attrition Rate by Job Role
-- Pinpoints specific roles that exhibit abnormally high attrition rates.
SELECT 
    jobrole,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY jobrole
ORDER BY attrition_rate_pct DESC;


-- Query 2.3: Attrition Rate by Education Field
-- Evaluates if specific professional backgrounds show different loyalty trends.
SELECT 
    educationfield,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY educationfield
ORDER BY attrition_rate_pct DESC;


-- =========================================================================
-- SECTION 3: SALARY & COMPENSATION ANALYSIS
-- Join demographic tables with performance/financial tables to analyze pay metrics.
-- =========================================================================

-- Query 3.1: Salary Statistics (Min, Max, Avg) by Department
-- Establishes baseline compensation ranges across departmental structures.
SELECT 
    t1.department,
    MIN(t2.monthlyincome) AS min_monthly_income,
    MAX(t2.monthlyincome) AS max_monthly_income,
    ROUND(AVG(t2.monthlyincome), 2) AS avg_monthly_income
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t1.department;


-- Query 3.2: Attrition Rate by Monthly Income Bins
-- Grouping monthly salary ranges to assess if lower-income brackets experience higher turnover.
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
GROUP BY 
    CASE 
        WHEN t2.monthlyincome < 3000 THEN 'Under $3k'
        WHEN t2.monthlyincome BETWEEN 3000 AND 6000 THEN '$3k - $6k'
        WHEN t2.monthlyincome BETWEEN 6001 AND 10000 THEN '$6k - $10k'
        WHEN t2.monthlyincome BETWEEN 10001 AND 15000 THEN '$10k - $15k'
        ELSE 'Over $15k'
    END
ORDER BY MIN(t2.monthlyincome) ASC;


-- Query 3.3: Attrition Rate by Percent Salary Hike Bins
-- Investigates if the scale of annual pay increases affects employee retention.
SELECT 
    CASE 
        WHEN t2.percentsalaryhike < 15 THEN 'Low Hike (<15%)'
        WHEN t2.percentsalaryhike BETWEEN 15 AND 20 THEN 'Medium Hike (15-20%)'
        ELSE 'High Hike (>20%)'
    END AS hike_category,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY 
    CASE 
        WHEN t2.percentsalaryhike < 15 THEN 'Low Hike (<15%)'
        WHEN t2.percentsalaryhike BETWEEN 15 AND 20 THEN 'Medium Hike (15-20%)'
        ELSE 'High Hike (>20%)'
    END
ORDER BY MIN(t2.percentsalaryhike) ASC;


-- =========================================================================
-- SECTION 4: OVERTIME ANALYSIS
-- Assess work intensity factors that drive burnout and eventual attrition.
-- =========================================================================

-- Query 4.1: Attrition Rate by Overtime Work Status
-- Direct comparison of employees who work overtime vs those who do not.
SELECT 
    t2.overtime,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t2.overtime;


-- Query 4.2: Attrition Rate of Employees Working Overtime with Stagnant Salary Hike (<15%)
-- Highlights a high-burnout subgroup: hard work (overtime) without adequate financial reward.
SELECT 
    t2.overtime,
    CASE WHEN t2.percentsalaryhike < 15 THEN 'Low Hike (<15%)' ELSE 'Normal/High Hike (15%+)' END AS hike_status,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY 
    t2.overtime,
    CASE WHEN t2.percentsalaryhike < 15 THEN 'Low Hike (<15%)' ELSE 'Normal/High Hike (15%+)' END
ORDER BY attrition_rate_pct DESC;


-- =========================================================================
-- SECTION 5: SATISFACTION ANALYSIS
-- Investigate self-reported employee satisfaction metrics.
-- =========================================================================

-- Query 5.1: Attrition Rate by Job Satisfaction Levels
-- Maps self-reported satisfaction to real attrition rates. (1=Low, 4=Very High)
SELECT 
    jobsatisfaction,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY jobsatisfaction
ORDER BY jobsatisfaction ASC;


-- Query 5.2: Attrition Rate by Work-Life Balance Score
-- Verifies the link between personal-professional balance and attrition. (1=Bad, 4=Best)
SELECT 
    t2.worklifebalance,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t2.worklifebalance
ORDER BY t2.worklifebalance ASC;


-- Query 5.3: Attrition Rate by Environment Satisfaction Score
-- Shows how workplace environment satisfaction influences attrition. (1=Low, 4=Very High)
SELECT 
    environmentsatisfaction,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1
GROUP BY environmentsatisfaction
ORDER BY environmentsatisfaction ASC;


-- =========================================================================
-- SECTION 6: TENURE & CAREER PROGRESSION ANALYSIS
-- Analyze time-based metrics and leadership relationships.
-- =========================================================================

-- Query 6.1: Attrition Rate by Years at the Company
-- Analyzes turnover based on organizational tenure brackets.
SELECT 
    CASE 
        WHEN t2.yearsatcompany <= 2 THEN 'New Joiner (0-2 yrs)'
        WHEN t2.yearsatcompany BETWEEN 3 AND 5 THEN 'Mid-Tenure (3-5 yrs)'
        WHEN t2.yearsatcompany BETWEEN 6 AND 10 THEN 'Tenured (6-10) yrs'
        ELSE 'Long-term (10+ yrs)'
    END AS tenure_bracket,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY 
    CASE 
        WHEN t2.yearsatcompany <= 2 THEN 'New Joiner (0-2 yrs)'
        WHEN t2.yearsatcompany BETWEEN 3 AND 5 THEN 'Mid-Tenure (3-5 yrs)'
        WHEN t2.yearsatcompany BETWEEN 6 AND 10 THEN 'Tenured (6-10) yrs'
        ELSE 'Long-term (10+ yrs)'
    END
ORDER BY MIN(t2.yearsatcompany) ASC;


-- Query 6.2: Attrition Rate by Years Since Last Promotion
-- Checks if stagnation in promotion increases likelihood of leaving.
SELECT 
    t2.yearssincelastpromotion,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t2.yearssincelastpromotion
ORDER BY t2.yearssincelastpromotion ASC;


-- Query 6.3: Attrition Rate by Years with Current Manager
-- Assesses the effect of management continuity and relationships on retention.
SELECT 
    t2.yearswithcurrmanager,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(SUM(CASE WHEN t1.attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS attrition_rate_pct
FROM hr_1 t1
JOIN hr_2 t2 ON t1.employeenumber = t2.employeeid
GROUP BY t2.yearswithcurrmanager
ORDER BY t2.yearswithcurrmanager ASC;
