import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from typing import Any, Dict, Optional, Tuple
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 150,
    max_depth: int = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 10,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Trains a Random Forest Classifier.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.

    Returns:
        RandomForestClassifier: Trained Random Forest model instance.
    """
    print(f"Training Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf


def train_xgboost(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42
) -> XGBClassifier:
    """
    Trains an XGBoost Classifier.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.

    Returns:
        XGBClassifier: Trained XGBoost model instance.
    """
    print(f"Training XGBoost Classifier (n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth})...")
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print("XGBoost training complete.")
    return xgb


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int = 42
) -> LogisticRegression:
    """
    Trains a baseline Logistic Regression Classifier.
    """
    print(f"Training Logistic Regression Classifier (C={C})...")
    lr = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    lr.fit(X_train, y_train)
    print("Logistic Regression training complete.")
    return lr


def save_model(model: Any, filepath: str) -> str:
    """
    Serializes and saves a trained model instance to disk using joblib.

    Args:
        model (Any): Trained classifier instance.
        filepath (str): Target save file path.

    Returns:
        str: Absolute path of saved model file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to: {filepath}")
    return filepath


def train_pipeline(
    model_type: str = "xgboost",
    save_artifact: bool = True,
    artifact_name: Optional[str] = None
) -> Tuple[Any, Any, Any, Any, Any, list]:
    """
    High-level training pipeline orchestrator.
    Ingests data, runs preprocessing, fits requested model, and saves model artifact.
    """
    from src.data_ingestion import ingest_data
    from src.preprocessing import preprocess_pipeline

    print(f"\n--- Initiating ML Model Training Pipeline [{model_type.upper()}] ---")
    raw_df = ingest_data()
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = preprocess_pipeline(raw_df)

    if model_type.lower() == "xgboost":
        model = train_xgboost(X_train, y_train)
        default_name = "xgboost_model.pkl"
    elif model_type.lower() in ["random_forest", "rf"]:
        model = train_random_forest(X_train, y_train)
        default_name = "random_forest_model.pkl"
    elif model_type.lower() in ["logistic_regression", "lr"]:
        model = train_logistic_regression(X_train, y_train)
        default_name = "logistic_regression_model.pkl"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if save_artifact:
        save_filename = artifact_name or default_name
        save_path = os.path.join(MODELS_DIR, save_filename)
        save_model(model, save_path)

    return model, X_test, y_test, scaler, encoders, feature_names



if __name__ == '__main__':
    print("=" * 60)
    print("Running Model Training Module Standalone Test")
    print("=" * 60)
    train_pipeline(model_type="xgboost")
