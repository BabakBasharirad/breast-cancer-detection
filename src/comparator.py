# =============================================================================
# comparator.py
# -----------------------------------------------------------------------------
# Comprehensive comparison of all balancing strategies.
# Produces complete results for both CV and test set evaluation,
# including all metrics, figures, and tables for the paper.
#
# Outputs:
#   - Data summary (dataset info for paper)
#   - Test set evaluation per strategy
#   - Confusion matrix per strategy
#   - ROC curve per strategy
#   - Precision-Recall curve per strategy
#   - Combined ROC curves (all strategies)
#   - Bar charts per metric (CV vs Test)
#   - Box plots of CV scores per strategy
#   - Summary tables (CSV)
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
    make_scorer,
)

from config import (
    RANDOM_STATE, CV_FOLDS, RF_PARAMS,
    METRICS_DIR, FIGURES_DIR, RESULTS_DIR,
)
from balancer import apply_balancing, get_class_weight_param, ALL_STRATEGIES
from model import build_random_forest
from utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PER_STRATEGY_DIR     = os.path.join(FIGURES_DIR, "per_strategy")
COMPARISON_DIR       = os.path.join(FIGURES_DIR, "comparison")
DATA_SUMMARY_PATH    = os.path.join(METRICS_DIR, "data_summary.json")
TEST_RESULTS_PATH    = os.path.join(METRICS_DIR, "test_results_all_strategies.csv")
CV_RESULTS_PATH      = os.path.join(METRICS_DIR, "cv_results_all_strategies.csv")
FULL_RESULTS_PATH    = os.path.join(METRICS_DIR, "full_results_all_strategies.csv")

# -----------------------------------------------------------------------------
# Plot Settings
# -----------------------------------------------------------------------------
PLOT_DPI    = 300
PLOT_FORMAT = "png"
PLOT_STYLE  = "whitegrid"
METRICS     = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy" : "Accuracy",
    "precision": "Precision",
    "recall"   : "Recall",
    "f1"       : "F1 Score",
    "roc_auc"  : "AUC-ROC",
}

STRATEGY_COLORS = {
    "none"         : "#7f8c8d",
    "class_weight" : "#2980b9",
    "random_over"  : "#27ae60",
    "random_under" : "#e74c3c",
    "smote"        : "#8e44ad",
    "smoteenn"     : "#f39c12",
    "smotetomek"   : "#16a085",
}

CV_SCORING = {
    "accuracy" : make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall"   : make_scorer(recall_score,    zero_division=0),
    "f1"       : make_scorer(f1_score,        zero_division=0),
    "roc_auc"  : "roc_auc",
}


# -----------------------------------------------------------------------------
# Data Summary
# -----------------------------------------------------------------------------

def save_data_summary(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_genes_raw: int,
    n_genes_filtered: int,
) -> dict:
    """
    Print and save dataset information for the paper.
    """
    summary = {
        "total_samples"         : int(len(y)),
        "total_tumor"           : int((y == 1).sum()),
        "total_normal"          : int((y == 0).sum()),
        "train_samples"         : int(len(y_train)),
        "train_tumor"           : int((y_train == 1).sum()),
        "train_normal"          : int((y_train == 0).sum()),
        "test_samples"          : int(len(y_test)),
        "test_tumor"            : int((y_test == 1).sum()),
        "test_normal"           : int((y_test == 0).sum()),
        "n_genes_before_filter" : int(n_genes_raw),
        "n_genes_after_filter"  : int(n_genes_filtered),
        "n_features_selected"   : int(X_train.shape[1]),
        "train_test_split"      : "80/20 stratified",
        "cv_folds"              : CV_FOLDS,
    }

    # Print to terminal
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY (for paper)")
    logger.info("=" * 60)
    logger.info(f"  Total samples          : {summary['total_samples']}")
    logger.info(f"  Tumor (positive)       : {summary['total_tumor']}")
    logger.info(f"  Normal (negative)      : {summary['total_normal']}")
    logger.info(f"  Imbalance ratio        : {summary['total_tumor'] / summary['total_normal']:.2f}:1")
    logger.info(f"  Train samples          : {summary['train_samples']} "
                f"(Tumor: {summary['train_tumor']}, Normal: {summary['train_normal']})")
    logger.info(f"  Test samples           : {summary['test_samples']} "
                f"(Tumor: {summary['test_tumor']}, Normal: {summary['test_normal']})")
    logger.info(f"  Genes before filtering : {summary['n_genes_before_filter']:,}")
    logger.info(f"  Genes after filtering  : {summary['n_genes_after_filter']:,}")
    logger.info(f"  Features selected      : {summary['n_features_selected']:,}")
    logger.info(f"  Train/Test split       : {summary['train_test_split']}")
    logger.info(f"  CV folds               : {summary['cv_folds']}")
    logger.info("=" * 60)

    ensure_dir(METRICS_DIR)
    with open(DATA_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Data summary saved to: {DATA_SUMMARY_PATH}")

    return summary


# -----------------------------------------------------------------------------
# Cross-Validation Per Strategy
# -----------------------------------------------------------------------------

def run_cv_all_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategies: list,
) -> tuple[pd.DataFrame, dict]:
    """
    Run cross-validation for all strategies.
    Returns summary DataFrame and raw fold scores dict.
    """
    logger.info("Running cross-validation for all strategies...")

    skf          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_summary   = []
    cv_raw_scores = {}

    for strategy in strategies:
        logger.info(f"  CV for strategy: {strategy}")
        X_bal, y_bal = apply_balancing(X_train, y_train, strategy)
        class_weight = get_class_weight_param(strategy)
        model        = build_random_forest(class_weight=class_weight)

        cv_results = cross_validate(
            model, X_bal, y_bal,
            cv=skf, scoring=CV_SCORING,
            return_train_score=False, n_jobs=-1,
        )

        row = {"strategy": strategy}
        fold_scores = {}
        for metric in METRICS:
            scores = cv_results[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(np.mean(scores)), 4)
            row[f"{metric}_std"]  = round(float(np.std(scores)),  4)
            fold_scores[metric]   = scores.tolist()

        cv_summary.append(row)
        cv_raw_scores[strategy] = fold_scores

    df_cv = pd.DataFrame(cv_summary).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    logger.info(f"CV results saved to: {CV_RESULTS_PATH}")

    return df_cv, cv_raw_scores


# -----------------------------------------------------------------------------
# Test Set Evaluation Per Strategy
# -----------------------------------------------------------------------------

def evaluate_all_strategies_on_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategies: list,
) -> tuple[pd.DataFrame, dict]:
    """
    Train a model per strategy and evaluate on test set.
    Returns summary DataFrame and per-strategy predictions dict.
    """
    logger.info("Evaluating all strategies on test set...")

    test_summary  = []
    predictions   = {}

    for strategy in strategies:
        logger.info(f"  Test evaluation for strategy: {strategy}")
        X_bal, y_bal = apply_balancing(X_train, y_train, strategy)
        class_weight = get_class_weight_param(strategy)
        model        = build_random_forest(class_weight=class_weight)
        model.fit(X_bal, y_bal)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        row = {
            "strategy" : strategy,
            "accuracy" : round(accuracy_score(y_test, y_pred),                   4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall"   : round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1"       : round(f1_score(y_test, y_pred, zero_division=0),        4),
            "roc_auc"  : round(roc_auc_score(y_test, y_prob),                    4),
        }
        test_summary.append(row)
        predictions[strategy] = {"y_pred": y_pred, "y_prob": y_prob, "model": model}

        logger.info(f"    AUC: {row['roc_auc']:.4f} | F1: {row['f1']:.4f} | "
                    f"Recall: {row['recall']:.4f} | Precision: {row['precision']:.4f}")

    df_test = pd.DataFrame(test_summary).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df_test.to_csv(TEST_RESULTS_PATH, index=False)
    logger.info(f"Test results saved to: {TEST_RESULTS_PATH}")

    return df_test, predictions


# -----------------------------------------------------------------------------
# Per-Strategy Figures
# -----------------------------------------------------------------------------

def plot_confusion_matrix_per_strategy(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Plot and save confusion matrix for each strategy."""
    ensure_dir(PER_STRATEGY_DIR)
    logger.info("Plotting confusion matrices per strategy...")

    for strategy, preds in predictions.items():
        cm     = confusion_matrix(y_test, preds["y_pred"])
        labels = ["Normal", "Tumor"]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.set_style(PLOT_STYLE)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Confusion Matrix\n{strategy}", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label",      fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        plt.tight_layout()

        path = os.path.join(PER_STRATEGY_DIR, f"confusion_matrix_{strategy}.png")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
        plt.close()

    logger.info(f"Confusion matrices saved to: {PER_STRATEGY_DIR}")


def plot_roc_curve_per_strategy(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Plot individual ROC curve for each strategy."""
    ensure_dir(PER_STRATEGY_DIR)
    logger.info("Plotting ROC curves per strategy...")

    for strategy, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc         = roc_auc_score(y_test, preds["y_prob"])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.set_style(PLOT_STYLE)
        ax.plot(fpr, tpr, color=STRATEGY_COLORS.get(strategy, "steelblue"),
                lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"ROC Curve — {strategy}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        path = os.path.join(PER_STRATEGY_DIR, f"roc_curve_{strategy}.png")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
        plt.close()

    logger.info(f"ROC curves saved to: {PER_STRATEGY_DIR}")


def plot_precision_recall_per_strategy(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Plot individual Precision-Recall curve for each strategy."""
    ensure_dir(PER_STRATEGY_DIR)
    logger.info("Plotting Precision-Recall curves per strategy...")

    for strategy, preds in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, preds["y_prob"])
        ap = average_precision_score(y_test, preds["y_prob"])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.set_style(PLOT_STYLE)
        ax.plot(recall, precision, color=STRATEGY_COLORS.get(strategy, "steelblue"),
                lw=2, label=f"AP = {ap:.4f}")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"Precision-Recall Curve — {strategy}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        path = os.path.join(PER_STRATEGY_DIR, f"pr_curve_{strategy}.png")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
        plt.close()

    logger.info(f"PR curves saved to: {PER_STRATEGY_DIR}")


# -----------------------------------------------------------------------------
# Aggregate Comparison Figures
# -----------------------------------------------------------------------------

def plot_combined_roc_curves(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Plot all strategies' ROC curves in one figure."""
    ensure_dir(COMPARISON_DIR)
    logger.info("Plotting combined ROC curves...")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.set_style(PLOT_STYLE)

    for strategy, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc         = roc_auc_score(y_test, preds["y_prob"])
        ax.plot(fpr, tpr, color=STRATEGY_COLORS.get(strategy, "gray"),
                lw=2, label=f"{strategy} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlim([0, 0.1]); ax.set_ylim([0.95, 1.001])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Balancing Strategies (Zoomed)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(COMPARISON_DIR, "roc_curves_all_strategies.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Combined ROC curves saved to: {path}")


def plot_combined_pr_curves(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Plot all strategies' Precision-Recall curves in one figure."""
    ensure_dir(COMPARISON_DIR)
    logger.info("Plotting combined Precision-Recall curves...")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.set_style(PLOT_STYLE)

    for strategy, preds in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, preds["y_prob"])
        ap = average_precision_score(y_test, preds["y_prob"])
        ax.plot(recall, precision, color=STRATEGY_COLORS.get(strategy, "gray"),
                lw=2, label=f"{strategy} (AP={ap:.3f})")

    ax.set_xlim([0.95, 1.001]); ax.set_ylim([0.95, 1.001])
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Balancing Strategies (Zoomed)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()

    path = os.path.join(COMPARISON_DIR, "pr_curves_all_strategies.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Combined PR curves saved to: {path}")


def plot_metric_bar_charts(
    df_cv: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """
    For each metric, plot a grouped bar chart comparing CV mean vs test score
    across all strategies.
    """
    ensure_dir(COMPARISON_DIR)
    logger.info("Plotting CV vs Test bar charts per metric...")

    strategies = df_test["strategy"].tolist()
    x          = np.arange(len(strategies))
    width      = 0.35

    for metric in METRICS:
        cv_means  = df_cv.set_index("strategy").loc[strategies, f"{metric}_mean"].values
        cv_stds   = df_cv.set_index("strategy").loc[strategies, f"{metric}_std"].values
        test_vals = df_test.set_index("strategy").loc[strategies, metric].values

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style(PLOT_STYLE)

        bars1 = ax.bar(x - width/2, cv_means, width, yerr=cv_stds, capsize=4,
                       label=f"CV Mean (±Std)", color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + width/2, test_vals, width,
                       label="Test Set", color="darkorange", alpha=0.85)

        # Annotate
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color="steelblue")
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color="darkorange")

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=10)
        ymin = max(0, min(np.min(cv_means - cv_stds), np.min(test_vals)) - 0.02)
        ax.set_ylim([ymin, 1.05])
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f"{METRIC_LABELS[metric]} — CV vs Test Set Comparison", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()

        path = os.path.join(COMPARISON_DIR, f"cv_vs_test_{metric}.png")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
        plt.close()

    logger.info(f"CV vs Test bar charts saved to: {COMPARISON_DIR}")


def plot_cv_boxplots(cv_raw_scores: dict) -> None:
    """
    Plot box plots of CV fold scores per metric per strategy.
    Shows score distribution and stability across folds.
    """
    ensure_dir(COMPARISON_DIR)
    logger.info("Plotting CV box plots...")

    for metric in METRICS:
        data       = []
        strategy_labels = []

        for strategy, scores in cv_raw_scores.items():
            data.append(scores[metric])
            strategy_labels.append(strategy)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style(PLOT_STYLE)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        colors = [STRATEGY_COLORS.get(s, "steelblue") for s in strategy_labels]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(strategy_labels) + 1))
        ax.set_xticklabels(strategy_labels, rotation=25, ha="right", fontsize=10)
        all_vals = [v for scores in data for v in scores]
        ymin = max(0, min(all_vals) - 0.02)
        ax.set_ylim([ymin, 1.02])
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f"{METRIC_LABELS[metric]} — CV Fold Distribution per Strategy",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(COMPARISON_DIR, f"boxplot_cv_{metric}.png")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
        plt.close()

    logger.info(f"CV box plots saved to: {COMPARISON_DIR}")


def plot_heatmap_comparison(df_test: pd.DataFrame) -> None:
    """
    Plot a heatmap of all metrics across all strategies on the test set.
    Good visual summary for the paper.
    """
    ensure_dir(COMPARISON_DIR)
    logger.info("Plotting metrics heatmap...")

    df_heat = df_test.set_index("strategy")[METRICS].rename(columns=METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.set_style("white")
    vmin = df_heat.values.min() - 0.005
    vmax = df_heat.values.max() + 0.001
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        vmin=vmin, vmax=vmax,
        annot_kws={"size": 10},
    )
    ax.set_title("Test Set Performance Heatmap — All Strategies",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Metric",   fontsize=11)
    ax.set_ylabel("Strategy", fontsize=11)
    plt.tight_layout()

    path = os.path.join(COMPARISON_DIR, "heatmap_all_strategies.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to: {path}")


def save_full_results_table(
    df_cv: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """
    Merge CV and test results into one comprehensive table and save.
    """
    df_cv_indexed   = df_cv.set_index("strategy")
    df_test_indexed = df_test.set_index("strategy")

    rows = []
    for strategy in df_test_indexed.index:
        row = {"strategy": strategy}
        for metric in METRICS:
            row[f"cv_{metric}_mean"] = df_cv_indexed.loc[strategy, f"{metric}_mean"]
            row[f"cv_{metric}_std"]  = df_cv_indexed.loc[strategy, f"{metric}_std"]
            row[f"test_{metric}"]    = df_test_indexed.loc[strategy, metric]
        rows.append(row)

    df_full = pd.DataFrame(rows)
    df_full.to_csv(FULL_RESULTS_PATH, index=False)
    logger.info(f"Full results table saved to: {FULL_RESULTS_PATH}")

    # Print nicely to terminal
    logger.info("\nFULL RESULTS SUMMARY:")
    logger.info("=" * 60)
    for _, row in df_full.iterrows():
        logger.info(f"\nStrategy: {row['strategy']}")
        for metric in METRICS:
            logger.info(
                f"  {METRIC_LABELS[metric]:12s} — "
                f"CV: {row[f'cv_{metric}_mean']:.4f} ± {row[f'cv_{metric}_std']:.4f} | "
                f"Test: {row[f'test_{metric}']:.4f}"
            )


# -----------------------------------------------------------------------------
# Full Comparison Pipeline
# -----------------------------------------------------------------------------

def run_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_genes_raw: int,
    n_genes_filtered: int,
    strategies: list = ALL_STRATEGIES,
) -> None:
    """
    Full comparison pipeline across all balancing strategies.

    Args:
        X, y               : Full dataset (for summary)
        X_train, y_train   : Training set
        X_test, y_test     : Test set
        n_genes_raw        : Number of genes before filtering
        n_genes_filtered   : Number of genes after filtering
        strategies         : List of balancing strategies to compare
    """
    logger.info("=" * 60)
    logger.info("COMPARATIVE STUDY: BALANCING STRATEGIES")
    logger.info("=" * 60)

    ensure_dir(PER_STRATEGY_DIR)
    ensure_dir(COMPARISON_DIR)

    # 1. Data summary
    save_data_summary(X, y, X_train, y_train, X_test, y_test,
                      n_genes_raw, n_genes_filtered)

    # 2. Cross-validation
    df_cv, cv_raw_scores = run_cv_all_strategies(X_train, y_train, strategies)

    # 3. Test set evaluation
    df_test, predictions = evaluate_all_strategies_on_test(
        X_train, y_train, X_test, y_test, strategies)

    # 4. Per-strategy figures
    plot_confusion_matrix_per_strategy(y_test, predictions)
    plot_roc_curve_per_strategy(y_test, predictions)
    plot_precision_recall_per_strategy(y_test, predictions)

    # 5. Aggregate comparison figures
    plot_combined_roc_curves(y_test, predictions)
    plot_combined_pr_curves(y_test, predictions)
    plot_metric_bar_charts(df_cv, df_test)
    plot_cv_boxplots(cv_raw_scores)
    plot_heatmap_comparison(df_test)

    # 6. Full results table
    save_full_results_table(df_cv, df_test)

    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETED")
    logger.info(f"  Per-strategy figures : {PER_STRATEGY_DIR}")
    logger.info(f"  Comparison figures   : {COMPARISON_DIR}")
    logger.info(f"  Metrics              : {METRICS_DIR}")
    logger.info("=" * 60)
