import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from typing import Any, Dict
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import RANDOM_STATE, build_preprocessing_steps, load_split, numeric_columns


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
CV_RESULTS_PATH = os.path.join(MODELS_DIR, "cv_results.pkl")

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "xgboost"]

# Metrics computed on every CV fold; "average_precision" is PR-AUC
CV_SCORING = {
    "pr_auc": "average_precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}


def make_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns an unfitted instance of each candidate classifier, keyed by name.
    Hyperparameters are fixed (no tuning).
    """
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=200, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        ),
    }


def build_pipeline(model: Any, numeric_cols: list, random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Wraps a classifier in the full imblearn Pipeline:
    scale numerics -> SMOTENC -> encode categoricals -> model.
    """
    return Pipeline(build_preprocessing_steps(numeric_cols, random_state) + [("model", model)])


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Stratified k-fold CV on the training set only. Every pipeline step,
    including SMOTENC, is refitted inside each training fold.

    Returns:
        Dict: {model_name: {metric: {"mean": float, "std": float}}}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    numeric_cols = numeric_columns(X_train)
    results = {}
    for name, model in make_models(random_state).items():
        print(f"Cross-validating {name} ({n_splits}-fold stratified)...")
        scores = cross_validate(build_pipeline(model, numeric_cols, random_state), X_train, y_train, cv=cv, scoring=CV_SCORING)
        results[name] = {
            metric: {"mean": float(np.mean(scores[f"test_{metric}"])), "std": float(np.std(scores[f"test_{metric}"]))}
            for metric in CV_SCORING
        }
        print(f"  PR-AUC {results[name]['pr_auc']['mean']:.4f} ± {results[name]['pr_auc']['std']:.4f} | "
              f"F1 {results[name]['f1']['mean']:.4f} ± {results[name]['f1']['std']:.4f}")
    return results


def select_headline(cv_results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Picks the model with the best mean CV PR-AUC; ties broken by mean CV F1."""
    return max(cv_results, key=lambda m: (cv_results[m]["pr_auc"]["mean"], cv_results[m]["f1"]["mean"]))


def save_model(model: Any, filepath: str) -> str:
    """
    Serializes and saves a fitted pipeline to disk using joblib.

    Args:
        model (Any): Fitted pipeline instance.
        filepath (str): Target save file path.

    Returns:
        str: Absolute path of saved model file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to: {filepath}")
    return filepath


def train_pipeline(n_splits: int = 5, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    High-level training orchestrator. Cross-validates every candidate on the
    training split, picks the headline model by CV PR-AUC, refits every
    pipeline on the full training split, and saves them to models/.
    The test split is not touched here.

    Returns:
        Dict[str, Any]: CV results, headline model name, and SMOTENC class counts.
    """
    print("\n--- Initiating ML Model Training Pipeline ---")
    X_train, _, y_train, _ = load_split()
    numeric_cols = numeric_columns(X_train)

    cv_results = cross_validate_models(X_train, y_train, n_splits=n_splits, random_state=random_state)
    headline = select_headline(cv_results)
    print(f"Headline model (best CV PR-AUC): {headline}")

    for name, model in make_models(random_state).items():
        pipeline = build_pipeline(model, numeric_cols, random_state)
        pipeline.fit(X_train, y_train)
        save_model(pipeline, model_path(name))

    # Re-run the scale -> SMOTENC steps alone to record the class balance the final fit trained on
    _, y_resampled = Pipeline(build_preprocessing_steps(numeric_cols, random_state)[:2]).fit_resample(X_train, y_train)

    summary = {
        "cv_results": cv_results,
        "headline_model": headline,
        "cv_folds": n_splits,
        "smotenc_class_counts": {
            "before": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "after": {str(k): int(v) for k, v in pd.Series(y_resampled).value_counts().sort_index().items()},
        },
    }
    save_model(summary, CV_RESULTS_PATH)
    return summary



if __name__ == '__main__':
    print("=" * 60)
    print("Running Model Training Module Standalone Test")
    print("=" * 60)
    train_pipeline()
