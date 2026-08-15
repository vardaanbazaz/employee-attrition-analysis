import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN



def clean_and_encode(
    df: pd.DataFrame, 
    target_col: str = "Attrition"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Cleans DataFrame, removes non-predictive IDs, encodes target variable to binary (1/0),
    and label-encodes categorical string columns.

    Args:
        df (pd.DataFrame): Raw merged HR DataFrame.
        target_col (str): Target column name representing flight risk.

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
            Features (X), Binary Target (y), and dictionary of fitted LabelEncoders.
    """
    df_clean = df.copy()

    # Drop non-predictive ID or constant columns if present
    cols_to_drop = ['EmployeeNumber', 'Employee ID', 'EmployeeCount', 'StandardHours', 'Over18']
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)

    # Encode Target Column (Yes -> 1, No -> 0)
    if target_col not in df_clean.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    if df_clean[target_col].dtype == object or isinstance(df_clean[target_col].iloc[0], str):
        target_series = (df_clean[target_col].astype(str).str.strip().str.lower() == 'yes').astype(int)
    else:
        target_series = df_clean[target_col].astype(int)

    features_df = df_clean.drop(columns=[target_col])

    # Encode categorical features
    encoders: Dict[str, LabelEncoder] = {}
    for col in features_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(features_df[col].astype(str))
        encoders[col] = le

    print(f"Encoded {len(encoders)} categorical features. Final feature shape: {features_df.shape}")
    return features_df, target_series, encoders


def apply_smote(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42,
    comb_smoteenn: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE (or SMOTEENN) oversampling to balance minority class samples.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        random_state (int): Random seed for reproducibility.
        comb_smoteenn (bool): Whether to use SMOTEENN (combined over/under sampling).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled X_train and y_train arrays.
    """
    print("Applying SMOTE oversampling to balance target class distribution...")

    if comb_smoteenn:
        sampler = SMOTEENN(random_state=random_state)
    else:
        sampler = SMOTE(random_state=random_state)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"Resampled training set shape: {X_resampled.shape}")
    print(f"Resampled target class balance: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    test_size: float = 0.3,
    random_state: int = 42,
    use_smote: bool = True,
    scale_features: bool = True
) -> Tuple[Any, Any, Any, Any, Optional[StandardScaler], Dict[str, LabelEncoder], list]:
    """
    Complete data preprocessing workflow pipeline: encoding, train/test split, scaling, and SMOTE.

    Returns:
        Tuple: (X_train_processed, X_test_processed, y_train_processed, y_test, scaler, encoders, feature_names)
    """
    X, y, encoders = clean_and_encode(df, target_col=target_col)
    feature_names = list(X.columns)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Apply SMOTE oversampling on training data
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

    return X_train, X_test, y_train, y_test, scaler, encoders, feature_names



if __name__ == '__main__':
    from src.data_ingestion import ingest_data
    print("=" * 60)
    print("Running Preprocessing Module Standalone Test")
    print("=" * 60)
    df = ingest_data()
    X_tr, X_te, y_tr, y_te, scaler, encs, feat_names = preprocess_pipeline(df)
    print(f"Preprocessing completed successfully!")
    print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
    print(f"X_test shape: {X_te.shape}, y_test shape: {y_te.shape}")
    print(f"Total features extracted: {len(feat_names)}")

