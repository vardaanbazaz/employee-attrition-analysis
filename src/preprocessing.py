import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


RANDOM_STATE = 42

DROP_COLS = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']

# Categorical columns, grouped by how they are encoded after resampling
ORDINAL_COLS = {'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']}
BINARY_COLS = ['Gender', 'OverTime']
NOMINAL_COLS = ['Department', 'JobRole', 'MaritalStatus', 'EducationField']
CATEGORICAL_COLS = list(ORDINAL_COLS) + BINARY_COLS + NOMINAL_COLS


def clean_data(
    df: pd.DataFrame,
    target_col: str = "Attrition"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drops non-predictive ID/constant columns and encodes the target variable to binary (1/0).
    Categorical features are left as strings; they are encoded inside the model pipeline.

    Args:
        df (pd.DataFrame): Raw HR DataFrame.
        target_col (str): Target column name representing flight risk.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and Binary Target (y).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if df_clean[target_col].dtype == object or isinstance(df_clean[target_col].iloc[0], str):
        target_series = (df_clean[target_col].astype(str).str.strip().str.lower() == 'yes').astype(int)
    else:
        target_series = df_clean[target_col].astype(int)

    features_df = df_clean.drop(columns=[target_col])
    print(f"Cleaned feature shape (before encoding): {features_df.shape}")
    return features_df, target_series


def numeric_columns(X: pd.DataFrame) -> List[str]:
    """Returns every feature column that is not one of the known categorical columns."""
    return [c for c in X.columns if c not in CATEGORICAL_COLS]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split. Deterministic, so train.py and evaluate.py
    reproduce the same 1,029 / 441 rows independently.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessing_steps(numeric_cols: List[str], random_state: int = RANDOM_STATE) -> list:
    """
    Builds the pre-model steps of the imblearn Pipeline:
    scale numerics -> SMOTENC (categoricals still unencoded) -> encode categoricals.

    SMOTENC only runs during fit, so at predict time the pipeline is scale -> encode -> model.

    Args:
        numeric_cols (List[str]): Columns to standardize.
        random_state (int): Seed for SMOTENC.

    Returns:
        list: [(name, step), ...] ready to extend with a ("model", estimator) step.
    """
    scale = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    smotenc = SMOTENC(categorical_features=CATEGORICAL_COLS, random_state=random_state)

    encode = ColumnTransformer(
        [
            ("ordinal", OrdinalEncoder(categories=list(ORDINAL_COLS.values())), list(ORDINAL_COLS)),
            ("binary", OneHotEncoder(drop='if_binary', sparse_output=False), BINARY_COLS),
            ("nominal", OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return [("scale", scale), ("smotenc", smotenc), ("encode", encode)]


def load_split(target_col: str = "Attrition") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Ingests the dataset, cleans it, and returns the fixed stratified 70/30 split.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    from src.data_ingestion import ingest_data
    X, y = clean_data(ingest_data(), target_col=target_col)
    return split_data(X, y)



if __name__ == '__main__':
    from imblearn.pipeline import Pipeline
    print("=" * 60)
    print("Running Preprocessing Module Standalone Test")
    print("=" * 60)
    X_tr, X_te, y_tr, y_te = load_split()
    prep = Pipeline(build_preprocessing_steps(numeric_columns(X_tr)))
    X_res, y_res = prep[:2].fit_resample(X_tr, y_tr)
    prep.fit(X_tr, y_tr)
    X_te_enc = prep.transform(X_te)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"SMOTENC class counts: {y_tr.value_counts().to_dict()} -> {pd.Series(y_res).value_counts().to_dict()}")
    print(f"Total features after encoding: {X_te_enc.shape[1]}")
