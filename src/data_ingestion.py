#!/usr/bin/env python3
"""
Data Ingestion Module for Employee Attrition & Predictive Analytics Platform.

Handles reading and validating the raw HR telemetry dataset
(WA_Fn-UseC_-HR-Employee-Attrition.csv — the IBM/Watson Analytics
"HR Employee Attrition" dataset, 1,470 rows x 35 columns, single file,
no merge required).
"""

import os
import pandas as pd

DEFAULT_DATA_DIR = "data/actual"
DEFAULT_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_raw_dataset(path: str = os.path.join(DEFAULT_DATA_DIR, DEFAULT_FILENAME)) -> pd.DataFrame:
    """
    Loads the raw HR attrition CSV dataset from the given path.

    Args:
        path (str): Path to the WA_Fn-UseC_-HR-Employee-Attrition.csv dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    print(f"Reading raw dataset: {path}")
    # utf-8-sig strips the file's leading BOM so the first column name
    # comes back as "Age", not "﻿Age".
    df = pd.read_csv(path, encoding="utf-8-sig")

    print(f"Loaded dataset. Total records: {len(df):,}, Total features: {df.shape[1]}")
    return df


def ingest_data(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    High-level execution function to load the HR attrition dataset from a directory.

    Args:
        data_dir (str): Relative or absolute path to raw data directory.

    Returns:
        pd.DataFrame: HR DataFrame ready for preprocessing.
    """
    csv_path = os.path.join(data_dir, DEFAULT_FILENAME)
    return load_raw_dataset(csv_path)


if __name__ == '__main__':
    print("=" * 60)
    print("Running Data Ingestion Module Standalone Test")
    print("=" * 60)
    df = ingest_data()
    print("\nDataset Preview:")
    print(df.head(3))
    print("\nTarget Class Distribution:")
    if 'Attrition' in df.columns:
        print(df['Attrition'].value_counts())
