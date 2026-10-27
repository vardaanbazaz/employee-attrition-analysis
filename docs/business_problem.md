# Business Problem: Employee Attrition & Retention

## Executive Summary
Employee attrition refers to the naturally occurring reduction of an organization's workforce when employees leave, retire, or resign and are not immediately replaced. High attrition can raise hiring and training costs, lose institutional knowledge, and affect team morale.

This project treats attrition as a predictive intelligence problem rather than a retrospective reporting one. It uses a modular Python machine learning pipeline to ingest IBM's fictional HR dataset, compare four classifiers (a dummy baseline, Logistic Regression, Random Forest and XGBoost), select one by cross-validation, and decode individual predictions with **SHAP (SHapley Additive exPlanations)** — moving HR teams from reactive exit surveys toward proactive, targeted retention interventions.

---

## The Cost of Employee Attrition
Employee turnover carries costs, which fall into:
* **Direct Costs**: Recruitment advertising, agency fees, hiring manager time, signing bonuses, and structured training materials for new hires.
* **Indirect Costs (Lost Productivity)**: Vacant roles mean other team members must assume extra workloads, causing burnout.
* **Loss of Institutional Knowledge**: When skilled employees depart, they take vital context, client relationships, and workflow expertise with them.

---

## Core Project Objectives
To address these challenges, this project seeks to answer several critical business questions:
1. **What is the organization's baseline attrition rate?** How does it vary across departments and job roles?
2. **What are the primary compensation-related attrition drivers?** Are employees leaving due to low monthly income or low daily rates?
3. **How do working conditions affect retention?** Is there a quantifiable relationship between overtime work, tenure, manager stability, and flight risk?
4. **Who are the high-risk employees?** Can individual employees be flagged for flight risk before they resign, with a transparent explanation of why?

---

## Business Scope
The scope of this project is focused on:
* **Descriptive Analytics**: Exploratory data analysis (EDA) over historical employee records to determine *what happened* — attrition rates by department and compensation bracket.
* **Diagnostic Analytics**: Using SHAP feature attribution to establish *why it happened* — for example, quantifying how overtime, low compensation, and short tenure push individual predictions toward higher flight risk.
* **Predictive Intelligence**: Training classification models (Logistic Regression, Random Forest, XGBoost) that flag at-risk employees ahead of resignation, giving HR leadership an individual-level, explainable intervention window.
