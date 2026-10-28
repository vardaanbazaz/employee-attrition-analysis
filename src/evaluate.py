import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from src.preprocessing import RANDOM_STATE, load_split
from src.train import CV_RESULTS_PATH, MODEL_NAMES, model_path

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
METRICS_PATH = os.path.join(ASSETS_DIR, "metrics.json")

# Dark-mode aesthetic (matches scripts/generate_visuals.py)
BG_COLOR = "#0B0F19"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "#1E293B"
TEAL_PRIMARY = "#14B8A6"
COPPER_PRIMARY = "#F97316"
VIOLET_ACCENT = "#A78BFA"

MODEL_COLORS = {
    "dummy": MUTED_TEXT,
    "logistic_regression": TEAL_PRIMARY,
    "random_forest": VIOLET_ACCENT,
    "xgboost": COPPER_PRIMARY,
}
MODEL_LABELS = {
    "dummy": "Dummy (most frequent)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def load_artifacts() -> Dict[str, Any]:
    """
    Loads the fitted pipelines and CV summary written by src/train.py.
    Nothing is retrained here.

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    paths = [CV_RESULTS_PATH] + [model_path(name) for name in MODEL_NAMES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts:\n  " + "\n  ".join(missing) + "\nRun `python src/train.py` first."
        )
    return {
        "summary": joblib.load(CV_RESULTS_PATH),
        "models": {name: joblib.load(model_path(name)) for name in MODEL_NAMES},
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Evaluates a fitted pipeline on the held-out test split and prints diagnostic metrics.

    Args:
        model (Any): Fitted imblearn Pipeline.
        X_test (pd.DataFrame): Raw (unencoded) test features.
        y_test (pd.Series): Ground truth binary test labels.
        model_name (str): Label identifying the model.

    Returns:
        Dict[str, Any]: Metric scores, confusion matrix, and predicted probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f" MODEL EVALUATION REPORT: {model_name.upper()}")
    print("=" * 60)
    print(f" Accuracy    : {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision   : {prec:.4f}")
    print(f" Recall      : {rec:.4f}")
    print(f" F1-Score    : {f1:.4f}")
    print(f" ROC-AUC     : {auc:.4f}")
    print(f" PR-AUC      : {pr_auc:.4f}")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   [[TN={cm[0][0]:<6}  FP={cm[0][1]:<6}]")
    print(f"    [FN={cm[1][0]:<6}  TP={cm[1][1]:<6}]]")
    print("-" * 60)
    print(" Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 60 + "\n")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "y_prob": y_prob,
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9.5)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _save(fig: plt.Figure, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")
    return output_path


def _legend(ax: plt.Axes, loc: str) -> None:
    legend = ax.legend(loc=loc, frameon=True, fontsize=9.5)
    legend.get_frame().set_facecolor(BG_COLOR)
    legend.get_frame().set_edgecolor(GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)


def plot_roc_curves(
    y_test: pd.Series,
    probs: Dict[str, np.ndarray],
    output_path: str = os.path.join(ASSETS_DIR, "roc_curves.png")
) -> str:
    """Overlays test-set ROC curves for every model."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], linestyle=":", color=GRID_COLOR, linewidth=1.2)
    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                linestyle="--" if name == "dummy" else "-",
                label=f"{MODEL_LABELS[name]} (AUC {roc_auc_score(y_test, y_prob):.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Held-out Test Set", fontsize=14, fontweight='bold', color=TEXT_COLOR, loc='left')
    _legend(ax, "lower right")
    return _save(fig, output_path)


def plot_pr_curves(
    y_test: pd.Series,
    probs: Dict[str, np.ndarray],
    output_path: str = os.path.join(ASSETS_DIR, "pr_curves.png")
) -> str:
    """
    Overlays test-set precision-recall curves for every model, with the prevalence (no-skill) line.
    The dummy model scores every row the same, so its "curve" is just the prevalence line;
    it is labelled on that line instead of drawn as a misleading interpolated diagonal.
    """
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_axes(ax)
    baseline_label = f"Prevalence ({prevalence:.3f})"
    if "dummy" in probs:
        baseline_label += f" = {MODEL_LABELS['dummy']} (AP {average_precision_score(y_test, probs['dummy']):.3f})"
    ax.axhline(prevalence, linestyle="--", color=MUTED_TEXT, linewidth=1.5, label=baseline_label)
    for name, y_prob in probs.items():
        if name == "dummy":
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(recall, precision, color=MODEL_COLORS[name], linewidth=2,
                label=f"{MODEL_LABELS[name]} (AP {average_precision_score(y_test, y_prob):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Held-out Test Set", fontsize=14, fontweight='bold', color=TEXT_COLOR, loc='left')
    _legend(ax, "upper right")
    return _save(fig, output_path)


def plot_confusion_matrix(
    cm: List[List[int]],
    model_name: str,
    output_path: str = os.path.join(ASSETS_DIR, "confusion_matrix.png")
) -> str:
    """Draws the test-set confusion matrix for the headline model."""
    cm_arr = np.array(cm)
    cmap = LinearSegmentedColormap.from_list("teal_dark", [BG_COLOR, TEAL_PRIMARY])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.imshow(cm_arr, cmap=cmap)
    labels = ["Retained (0)", "Attrition (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.set_xlabel("Predicted", color=MUTED_TEXT)
    ax.set_ylabel("Actual", color=MUTED_TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    threshold = cm_arr.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha='center', va='center', fontsize=18, fontweight='bold',
                    color=BG_COLOR if cm_arr[i, j] > threshold else TEXT_COLOR)
    ax.set_title(f"Confusion Matrix — {MODEL_LABELS[model_name]} (test set)", fontsize=13, fontweight='bold', color=TEXT_COLOR, loc='left')
    return _save(fig, output_path)


def generate_shap_explainability(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    output_path: str = os.path.join(ASSETS_DIR, "shap_summary.png"),
    max_samples: int = 500,
    top_n: int = 10,
    random_state: int = RANDOM_STATE
) -> List[Dict[str, Any]]:
    """
    Computes SHAP values for the pipeline's final model on the encoded test set
    and exports a summary plot. Uses LinearExplainer for Logistic Regression and
    TreeExplainer for tree models.

    Args:
        pipeline (Any): Fitted imblearn Pipeline (scale -> SMOTENC -> encode -> model).
        X_train (pd.DataFrame): Raw training features (LinearExplainer background).
        X_test (pd.DataFrame): Raw test features to explain.
        model_name (str): MODEL_LABELS key of the explained model, used in the plot title.
        output_path (str): File destination path for SHAP plot artifact.
        max_samples (int): Sample size cap to accelerate SHAP computation.
        top_n (int): Number of top features to return.

    Returns:
        List[Dict[str, Any]]: Top features by mean |SHAP value|.
    """
    print("Calculating SHAP values for model explainability...")
    preprocess = pipeline[:-1]  # SMOTENC is skipped outside of fit
    model = pipeline.named_steps["model"]

    X_explain = preprocess.transform(X_test)
    if len(X_explain) > max_samples:
        X_explain = X_explain.sample(n=max_samples, random_state=random_state)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, preprocess.transform(X_train))
        explainer_name = "LinearExplainer"
    elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        explainer_name = "TreeExplainer"
    else:
        raise ValueError(f"No SHAP explainer configured for {type(model).__name__}")

    shap_vals = explainer.shap_values(X_explain)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if np.ndim(shap_vals) == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_explain.columns).sort_values(ascending=False)
    top_features = [{"feature": f, "mean_abs_shap": float(v)} for f, v in mean_abs.head(top_n).items()]

    plt.close('all')
    shap.summary_plot(shap_vals, features=X_explain, show=False, plot_type="dot", color_bar=True)

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

    top3 = ", ".join(f["feature"] for f in top_features[:3])
    fig.suptitle(f"SHAP Summary — {MODEL_LABELS[model_name]} (test set)", fontsize=15, fontweight='bold', color=TEXT_COLOR, x=0.125, y=0.98, ha='left')
    fig.text(0.125, 0.94, f"{explainer_name}, mean |SHAP| ranking. Top features: {top3}", fontsize=10.5, color=MUTED_TEXT)

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    plt.close('all')

    print(f"SHAP summary plot successfully saved to: {output_path}")
    return top_features


def print_results_table(models_report: Dict[str, Any], headline: str) -> None:
    """Prints CV (mean ± std) and test metrics side by side for every model."""
    metrics = ["pr_auc", "f1", "roc_auc", "precision", "recall", "accuracy"]
    print("\n" + "=" * 60)
    print(" CV (5-fold, train only) vs TEST — headline marked *")
    print("=" * 60)
    for name, report in models_report.items():
        print(f"{'*' if name == headline else ' '} {MODEL_LABELS[name]}")
        for m in metrics:
            cv = report["cv"][m]
            print(f"    {m:<10} CV {cv['mean']:.4f} ± {cv['std']:.4f}   test {report['test'][m]:.4f}")
    print("=" * 60 + "\n")


def run_full_pipeline_evaluation(generate_shap: bool = True) -> Dict[str, Any]:
    """
    Loads the pipelines saved by src/train.py, evaluates each on the held-out
    test split, writes ROC/PR/confusion/SHAP plots and assets/metrics.json.
    The test split is used for reporting only; the headline model was chosen by CV in train.py.
    """
    artifacts = load_artifacts()
    summary = artifacts["summary"]
    headline = summary["headline_model"]
    X_train, X_test, y_train, y_test = load_split()

    models_report = {}
    probs = {}
    for name, pipeline in artifacts["models"].items():
        test_metrics = evaluate_model(pipeline, X_test, y_test, model_name=MODEL_LABELS[name])
        probs[name] = test_metrics.pop("y_prob")
        models_report[name] = {"cv": summary["cv_results"][name], "test": test_metrics}

    plot_roc_curves(y_test, probs)
    plot_pr_curves(y_test, probs)
    plot_confusion_matrix(models_report[headline]["test"]["confusion_matrix"], headline)

    top_features = []
    if generate_shap:
        top_features = generate_shap_explainability(artifacts["models"][headline], X_train, X_test, headline)

    n_features = artifacts["models"][headline][:-1].transform(X_test.head(1)).shape[1]
    metrics = {
        "random_state": RANDOM_STATE,
        "split": {"train_rows": len(X_train), "test_rows": len(X_test), "test_size": 0.3, "stratified": True},
        "prevalence": {
            "overall": float(pd.concat([y_train, y_test]).mean()),
            "train": float(y_train.mean()),
            "test": float(y_test.mean()),
        },
        "n_features_raw": X_train.shape[1],
        "n_features_encoded": int(n_features),
        "smotenc_class_counts": summary["smotenc_class_counts"],
        "selection": {
            "cv": f"StratifiedKFold(n_splits={summary['cv_folds']}, shuffle=True, random_state={RANDOM_STATE}) on training split only",
            "primary_metric": "pr_auc (average precision)",
            "tiebreak_metric": "f1",
        },
        "headline_model": headline,
        "models": models_report,
        "top_shap_features": top_features,
    }

    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"Metrics written to: {METRICS_PATH}")

    print_results_table(models_report, headline)
    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("Running Model Evaluation & SHAP Explainability Standalone Test")
    print("=" * 60)
    run_full_pipeline_evaluation(generate_shap=True)
