# =============================================================================
# evaluator.py
# -----------------------------------------------------------------------------
# Evaluates the final trained model on the held-out test set and generates
# all evaluation outputs required for the paper:
#
#   1. Classification metrics  — accuracy, precision, recall, F1, AUC-ROC
#   2. Confusion matrix plot
#   3. ROC curve plot
#   4. Balancing strategy comparison plot
#   5. Metrics saved to JSON
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from config import (
    METRICS_OUTPUT_FILE,
    ROC_CURVE_FILE,
    CONFUSION_MATRIX_FILE,
    FIGURES_DIR,
    METRICS_DIR,
)
from utils import setup_logger, ensure_dir, save_metrics

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Plot Paths
# -----------------------------------------------------------------------------
STRATEGY_COMPARISON_FILE = os.path.join(FIGURES_DIR, "strategy_comparison.png")

# -----------------------------------------------------------------------------
# Plot Style
# -----------------------------------------------------------------------------
PLOT_STYLE  = "whitegrid"
PLOT_DPI    = 300
PLOT_FORMAT = "png"


# -----------------------------------------------------------------------------
# Classification Metrics
# -----------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute all classification metrics for the test set.

    Args:
        y_true : True binary labels.
        y_pred : Predicted binary labels.
        y_prob : Predicted probabilities for the positive class.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred),                          4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),        4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0),           4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0),               4),
        "roc_auc"  : round(roc_auc_score(y_true, y_prob),                           4),
        "n_test_samples"  : int(len(y_true)),
        "n_positive_test" : int(y_true.sum()),
        "n_negative_test" : int((y_true == 0).sum()),
    }

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1 Score  : {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC   : {metrics['roc_auc']:.4f}")
    logger.info("=" * 60)

    # Full classification report
    report = classification_report(y_true, y_pred, target_names=["Normal", "Tumor"])
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


# -----------------------------------------------------------------------------
# Confusion Matrix Plot
# -----------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: str = CONFUSION_MATRIX_FILE,
) -> None:
    """
    Plot and save a styled confusion matrix heatmap.

    Args:
        y_true    : True binary labels.
        y_pred    : Predicted binary labels.
        save_path : File path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    sns.set_style(PLOT_STYLE)

    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Normal (0)", "Tumor (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to: {save_path}")


# -----------------------------------------------------------------------------
# ROC Curve Plot
# -----------------------------------------------------------------------------

def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    auc_score: float,
    save_path: str = ROC_CURVE_FILE,
) -> None:
    """
    Plot and save the ROC curve for the final model on the test set.

    Args:
        y_true    : True binary labels.
        y_prob    : Predicted probabilities for the positive class.
        auc_score : Pre-computed AUC-ROC score.
        save_path : File path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    sns.set_style(PLOT_STYLE)

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr, tpr,
        color="steelblue",
        lw=2,
        label=f"Random Forest (AUC = {auc_score:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved to: {save_path}")


# -----------------------------------------------------------------------------
# Balancing Strategy Comparison Plot
# -----------------------------------------------------------------------------

def plot_strategy_comparison(
    cv_results: pd.DataFrame,
    save_path: str = STRATEGY_COMPARISON_FILE,
) -> None:
    """
    Plot a grouped bar chart comparing AUC-ROC and F1 scores across
    all balancing strategies from cross-validation results.

    Args:
        cv_results : DataFrame from model.compare_balancing_strategies().
        save_path  : File path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    sns.set_style(PLOT_STYLE)

    strategies = cv_results["strategy"]
    auc_means  = cv_results["roc_auc_mean"]
    auc_stds   = cv_results["roc_auc_std"]
    f1_means   = cv_results["f1_mean"]
    f1_stds    = cv_results["f1_std"]

    x     = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        x - width / 2, auc_means, width,
        yerr=auc_stds, capsize=4,
        label="AUC-ROC", color="steelblue", alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, f1_means, width,
        yerr=f1_stds, capsize=4,
        label="F1 Score", color="darkorange", alpha=0.85
    )

    # Annotate bars with mean values
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8, color="steelblue"
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8, color="darkorange"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=10)
    ax.set_ylim([0.0, 1.10])
    ax.set_ylabel("Score (Mean ± Std)", fontsize=12)
    ax.set_title(
        f"Balancing Strategy Comparison ({CV_FOLDS}-Fold Cross-Validation)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Highlight best strategy
    best_idx = cv_results["roc_auc_mean"].idxmax()
    ax.get_xticklabels()[best_idx].set_color("green")
    ax.get_xticklabels()[best_idx].set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()

    logger.info(f"Strategy comparison plot saved to: {save_path}")


# -----------------------------------------------------------------------------
# Full Evaluation Pipeline
# -----------------------------------------------------------------------------

def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: pd.DataFrame,
    best_strategy: str,
) -> dict:
    """
    Full evaluation pipeline for the final model on the test set.

    Steps:
        1. Generate predictions and probabilities
        2. Compute classification metrics
        3. Plot confusion matrix
        4. Plot ROC curve
        5. Plot balancing strategy comparison
        6. Save all metrics to JSON

    Args:
        model         : Trained final RandomForestClassifier.
        X_test        : Test feature matrix.
        y_test        : Test labels.
        cv_results    : CV comparison DataFrame from model pipeline.
        best_strategy : Name of the best balancing strategy.

    Returns:
        Dictionary of test set evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 60)

    ensure_dir(FIGURES_DIR)
    ensure_dir(METRICS_DIR)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Step 1: Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["best_strategy"] = best_strategy

    # Step 2: Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Step 3: ROC curve
    plot_roc_curve(y_test, y_prob, metrics["roc_auc"])

    # Step 4: Strategy comparison
    plot_strategy_comparison(cv_results)

    # Step 5: Save metrics
    save_metrics(metrics, METRICS_OUTPUT_FILE)
    logger.info(f"Evaluation metrics saved to: {METRICS_OUTPUT_FILE}")

    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 60)

    return metrics


# Import CV_FOLDS for plot title
from config import CV_FOLDS
