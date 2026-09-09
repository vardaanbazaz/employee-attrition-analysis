-- ==========================================
-- DATABASE SCHEMA CREATION
-- Project: Employee Attrition & Performance Analysis
-- Database: MySQL
-- ==========================================

CREATE DATABASE IF NOT EXISTS hrdataanalytics;
USE hrdataanalytics;

-- ------------------------------------------
-- Table 1: hr_1 (Employee Demographics and Job Roles)
-- Contains demographic info, job characteristics, and core satisfaction metrics.
-- ------------------------------------------
DROP TABLE IF EXISTS hr_1;
CREATE TABLE hr_1 (
    EmployeeNumber          INT PRIMARY KEY,
    Age                    INT,
    Attrition              VARCHAR(5),  -- "Yes" or "No"
    BusinessTravel         VARCHAR(30), -- "Travel_Rarely", "Travel_Frequently", "Non-Travel"
    DailyRate              INT,
    Department             VARCHAR(50),
    DistanceFromHome       INT,
    Education              INT,         -- 1 (Below College) to 5 (Doctor)
    EducationField         VARCHAR(50),
    EmployeeCount           INT,
    EnvironmentSatisfaction INT,         -- 1 (Low) to 4 (Very High)
    Gender                 VARCHAR(10),
    HourlyRate             INT,
    JobInvolvement         INT,         -- 1 (Low) to 4 (Very High)
    JobLevel                INT,         -- 1 (Entry) to 5 (Executive)
    JobRole                VARCHAR(50),
    JobSatisfaction        INT,         -- 1 (Low) to 4 (Very High)
    MaritalStatus          VARCHAR(20)  -- "Single", "Married", "Divorced"
);

-- ------------------------------------------
-- Table 2: hr_2 (Employee Performance, Compensation, and Tenure)
-- Contains financial information, promotion history, and performance metrics.
-- Maps 1-to-1 with hr_1 via EmployeeID.
-- ------------------------------------------
DROP TABLE IF EXISTS hr_2;
CREATE TABLE hr_2 (
    EmployeeID             INT PRIMARY KEY,
    MonthlyIncome          INT,
    MonthlyRate            INT,
    NumCompaniesWorked     INT,
    Over18                 VARCHAR(5),  -- "Y" or "N"
    OverTime               VARCHAR(5),  -- "Yes" or "No"
    PerformanceRating      INT,         -- 1 (Low) to 4 (Outstanding)
    RelationshipSatisfaction INT,        -- 1 (Low) to 4 (Very High)
    StandardHours          INT,
    StockOptionLevel       INT,         -- 0 (None) to 3 (High)
    WorkLifeBalance        INT,         -- 1 (Bad) to 4 (Best)
    PercentSalaryHike      INT,
    TotalWorkingYears      INT,
    TrainingTimesLastYear  INT,
    YearsAtCompany         INT,
    YearsInCurrentRole     INT,
    YearsSinceLastPromotion INT,
    YearsWithCurrManager   INT,
    FOREIGN KEY (EmployeeID) REFERENCES hr_1(EmployeeNumber) ON DELETE CASCADE
);
