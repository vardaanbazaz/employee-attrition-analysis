#!/usr/bin/env python3
"""
Data Ingestion Module for Employee Attrition & Predictive Analytics Platform.

Handles reading, validating, and merging raw HR telemetry datasets
(HR_1.xlsx demographics/satisfaction and HR_2.xlsx compensation/tenure).
"""

import os
from typing import Tuple, Optional
import pandas as pd


def load_raw_datasets(
    hr1_path: str = "data/raw/HR_1.xlsx",
    hr2_path: str = "data/raw/HR_2.xlsx"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw Excel datasets HR_1 and HR_2 from the given paths.

    Args:
        hr1_path (str): Path to HR_1.xlsx dataset.
        hr2_path (str): Path to HR_2.xlsx dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded DataFrames for HR_1 and HR_2.
    """
    if not os.path.exists(hr1_path):
        raise FileNotFoundError(f"Raw dataset not found at: {hr1_path}")
    if not os.path.exists(hr2_path):
        raise FileNotFoundError(f"Raw dataset not found at: {hr2_path}")

    print(f"Reading raw dataset 1: {hr1_path}")
    df1 = pd.read_excel(hr1_path)
    
    print(f"Reading raw dataset 2: {hr2_path}")
    df2 = pd.read_excel(hr2_path)

    return df1, df2


def merge_hr_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges HR_1 and HR_2 DataFrames on Employee Number / ID keys.

    Args:
        df1 (pd.DataFrame): Demographics and satisfaction DataFrame (HR_1).
        df2 (pd.DataFrame): Compensation and tenure DataFrame (HR_2).

    Returns:
        pd.DataFrame: Clean, merged tabular DataFrame.
    """
    # Verify primary key presence
    if 'EmployeeNumber' not in df1.columns:
        raise KeyError("Expected 'EmployeeNumber' in HR_1 dataset.")
    if 'Employee ID' not in df2.columns:
        raise KeyError("Expected 'Employee ID' in HR_2 dataset.")

    # Execute inner join on primary key
    df_merged = df1.merge(df2, left_on='EmployeeNumber', right_on='Employee ID')

    # Drop trailing unnamed or duplicate columns if present
    unnamed_cols = [c for c in df_merged.columns if 'Unnamed' in c]
    if unnamed_cols:
        df_merged.drop(columns=unnamed_cols, inplace=True)

    print(f"Successfully merged datasets. Total records: {len(df_merged):,}, Total features: {df_merged.shape[1]}")
    return df_merged


def ingest_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    High-level execution function to load and merge HR datasets from a directory.

    Args:
        data_dir (str): Relative or absolute path to raw data directory.

    Returns:
        pd.DataFrame: Merged HR DataFrame ready for preprocessing.
    """
    hr1_path = os.path.join(data_dir, "HR_1.xlsx")
    hr2_path = os.path.join(data_dir, "HR_2.xlsx")
    df1, df2 = load_raw_datasets(hr1_path, hr2_path)
    return merge_hr_datasets(df1, df2)


if __name__ == '__main__':
    print("=" * 60)
    print("Running Data Ingestion Module Standalone Test")
    print("=" * 60)
    merged_df = ingest_data()
    print("\nDataset Preview:")
    print(merged_df.head(3))
    print("\nTarget Class Distribution:")
    if 'Attrition' in merged_df.columns:
        print(merged_df['Attrition'].value_counts())
