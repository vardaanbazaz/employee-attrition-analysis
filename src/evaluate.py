import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))


def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test dataset and prints comprehensive diagnostic metrics.

    Args:
        model (Any): Trained scikit-learn or XGBoost classifier.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Ground truth binary test labels.
        model_name (str): Label identifying the model.

    Returns:
        Dict[str, Any]: Dictionary containing metric scores and classification report.
    """
    y_pred = model.predict(X_test)
    
    # Calculate probability scores for ROC-AUC if model supports predict_proba
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": clf_report
    }

    print("\n" + "=" * 60)
    print(f" MODEL EVALUATION REPORT: {model_name.upper()}")
    print("=" * 60)
    print(f" Accuracy    : {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision   : {prec:.4f}")
    print(f" Recall      : {rec:.4f}")
    print(f" F1-Score    : {f1:.4f}")
    print(f" ROC-AUC     : {auc:.4f}")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:<6}  FP={cm[0][1]:<6}]")
    print(f"    [FN={cm[1][0]:<6}  TP={cm[1][1]:<6}]]")
    print("-" * 60)
    print(" Detailed Classification Report:")
    print(clf_report)
    print("=" * 60 + "\n")

    return metrics


def generate_shap_explainability(
    model: Any, 
    X_test: np.ndarray, 
    feature_names: list,
    output_path: str = os.path.join(ASSETS_DIR, "shap_summary.png"),
    max_samples: int = 500
) -> str:
    """
    Computes SHAP (SHapley Additive exPlanations) values for the trained classifier
    and exports a high-resolution summary plot saved to assets/shap_summary.png.

    Args:
        model (Any): Trained classifier (XGBoost or Random Forest).
        X_test (np.ndarray): Test dataset feature array.
        feature_names (list): List of feature column names.
        output_path (str): File destination path for SHAP plot artifact.
        max_samples (int): Sample size to accelerate SHAP computation.

    Returns:
        str: Path of saved SHAP summary plot.
    """
    print("Calculating SHAP values for model explainability...")

    # Subsample test data for speed if large
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), size=max_samples, replace=False)
        X_sub = X_test[indices]
    else:
        X_sub = X_test

    X_sub_df = pd.DataFrame(X_sub, columns=feature_names)

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_vals = explainer.shap_values(X_sub_df)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Configure dark-mode aesthetic
    BG_COLOR = "#0B0F19"
    TEXT_COLOR = "#F8FAFC"
    MUTED_TEXT = "#94A3B8"
    GRID_COLOR = "#1E293B"

    plt.close('all')
    fig = plt.figure(figsize=(11, 8), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)

    # Generate SHAP summary dot plot
    shap.summary_plot(
        shap_vals,
        features=X_sub_df,
        show=False,
        plot_type="dot",
        color_bar=True
    )

    # Polish axes, text, and title for high contrast
    fig = plt.gcf()
    fig.patch.set_facecolor(BG_COLOR)

    for ax in fig.axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=MUTED_TEXT, labelsize=9.5)
        ax.xaxis.label.set_color(MUTED_TEXT)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for t in ax.get_yticklabels():
            t.set_color(TEXT_COLOR)
            t.set_fontsize(10)
            t.set_fontweight('bold')

    fig.suptitle("SHAP Feature Importance & Flight Risk Summary Plot", fontsize=15, fontweight='bold', color=TEXT_COLOR, x=0.125, y=0.98, ha='left')
    fig.text(0.125, 0.94, "Quantifying directional feature impacts (Overtime, Income, Tenure) on churn probability", fontsize=10.5, color=MUTED_TEXT)

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    plt.close('all')

    print(f"SHAP summary plot successfully saved to: {output_path}")
    return output_path




def run_full_pipeline_evaluation(model_type: str = "xgboost", generate_shap: bool = True) -> Dict[str, Any]:
    """
    Executes end-to-end ingestion, preprocessing, training, evaluation, and SHAP explainability.
    """
    from src.train import train_pipeline
    model, X_test, y_test, scaler, encoders, feature_names = train_pipeline(model_type=model_type, save_artifact=True)
    metrics = evaluate_model(model, X_test, y_test, model_name=f"{model_type.title()} Classifier")

    if generate_shap:
        generate_shap_explainability(model, X_test, feature_names)

    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("Running Model Evaluation & SHAP Explainability Standalone Test")
    print("=" * 60)
    run_full_pipeline_evaluation(model_type="xgboost", generate_shap=True)
