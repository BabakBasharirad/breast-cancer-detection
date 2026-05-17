# =============================================================================
# explainer.py
# -----------------------------------------------------------------------------
# Model explainability module. Generates feature importance analysis using
# three complementary methods:
#
#   1. MDI (Mean Decrease in Impurity)  — built-in RF feature importance
#   2. Permutation Importance           — model-agnostic, more reliable
#   3. SHAP Values                      — state-of-the-art explainability
#
# All outputs are saved as publication-ready figures and CSV files.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from config import (
    FIGURES_DIR,
    METRICS_DIR,
    RESULTS_DIR,
    N_TOP_IMPORTANT_FEATURES,
    RANDOM_STATE,
)
from utils import setup_logger, ensure_dir, load_gene_name_map, apply_gene_names

GENE_NAME_MAP_PATH = os.path.join(RESULTS_DIR, "metrics", "gene_name_mapping.json")

def _get_gene_map():
    """Load gene name map, return empty dict if not found."""
    try:
        return load_gene_name_map(GENE_NAME_MAP_PATH)
    except FileNotFoundError:
        return {}

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
MDI_IMPORTANCE_FILE        = os.path.join(FIGURES_DIR, "feature_importance_mdi.png")
PERMUTATION_IMPORTANCE_FILE= os.path.join(FIGURES_DIR, "feature_importance_permutation.png")
SHAP_SUMMARY_FILE          = os.path.join(FIGURES_DIR, "shap_summary.png")
SHAP_BAR_FILE              = os.path.join(FIGURES_DIR, "shap_bar.png")
MDI_CSV_FILE               = os.path.join(METRICS_DIR,  "feature_importance_mdi.csv")
PERMUTATION_CSV_FILE       = os.path.join(METRICS_DIR,  "feature_importance_permutation.csv")
SHAP_CSV_FILE              = os.path.join(METRICS_DIR,  "feature_importance_shap.csv")

# -----------------------------------------------------------------------------
# Plot Style
# -----------------------------------------------------------------------------
PLOT_DPI    = 300
PLOT_FORMAT = "png"
PLOT_STYLE  = "whitegrid"


# -----------------------------------------------------------------------------
# 1. MDI Feature Importance
# -----------------------------------------------------------------------------

def compute_mdi_importance(
    model: RandomForestClassifier,
    feature_names: list,
    n_top: int = N_TOP_IMPORTANT_FEATURES,
) -> pd.DataFrame:
    """
    Compute Mean Decrease in Impurity (MDI) feature importance from the
    trained Random Forest. This is the built-in feature importance based
    on how much each gene reduces impurity across all trees.

    Note: MDI can be biased toward high-cardinality features. Use alongside
    permutation importance and SHAP for a complete picture.

    Args:
        model        : Trained RandomForestClassifier.
        feature_names: List of gene names (columns of X_train).
        n_top        : Number of top features to plot.

    Returns:
        DataFrame of all genes with their MDI importance scores, sorted descending.
    """
    logger.info("Computing MDI (Mean Decrease in Impurity) feature importance...")

    gene_map    = _get_gene_map()
    importances = model.feature_importances_
    std         = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )

    df = pd.DataFrame({
        "gene"      : apply_gene_names(feature_names, gene_map),
        "gene_id"   : feature_names,
        "importance": importances,
        "std"       : std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save full importance table
    ensure_dir(METRICS_DIR)
    df.to_csv(MDI_CSV_FILE, index=False)
    logger.info(f"MDI importance saved to: {MDI_CSV_FILE}")

    # Plot top N
    _plot_importance(
        df.head(n_top),
        importance_col="importance",
        std_col="std",
        title=f"Top {n_top} Genes — MDI Feature Importance",
        xlabel="Mean Decrease in Impurity",
        color="steelblue",
        save_path=MDI_IMPORTANCE_FILE,
    )

    logger.info(f"Top 5 genes by MDI: {df['gene'].head(5).tolist()}")
    return df


# -----------------------------------------------------------------------------
# 2. Permutation Importance
# -----------------------------------------------------------------------------

def compute_permutation_importance(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_top: int = N_TOP_IMPORTANT_FEATURES,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute permutation importance on the test set. For each gene, its values
    are randomly shuffled and the drop in model performance (AUC-ROC) is
    measured. Genes that cause a large performance drop are most important.

    This method is more reliable than MDI as it is evaluated on unseen data
    and is not biased toward high-cardinality features.

    Args:
        model     : Trained RandomForestClassifier.
        X_test    : Test feature matrix.
        y_test    : Test labels.
        n_top     : Number of top features to plot.
        n_repeats : Number of times to permute each feature.

    Returns:
        DataFrame of all genes with their permutation importance scores.
    """
    logger.info("Computing permutation importance on test set...")
    logger.info(f"  n_repeats: {n_repeats} (this may take a few minutes...)")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    gene_map = _get_gene_map()
    df = pd.DataFrame({
        "gene"      : apply_gene_names(X_test.columns.tolist(), gene_map),
        "gene_id"   : X_test.columns.tolist(),
        "importance": result.importances_mean,
        "std"       : result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save full importance table
    ensure_dir(METRICS_DIR)
    df.to_csv(PERMUTATION_CSV_FILE, index=False)
    logger.info(f"Permutation importance saved to: {PERMUTATION_CSV_FILE}")

    # Plot top N
    _plot_importance(
        df.head(n_top),
        importance_col="importance",
        std_col="std",
        title=f"Top {n_top} Genes — Permutation Importance (Test Set)",
        xlabel="Mean Decrease in AUC-ROC",
        color="darkorange",
        save_path=PERMUTATION_IMPORTANCE_FILE,
    )

    logger.info(f"Top 5 genes by permutation: {df['gene'].head(5).tolist()}")
    return df


# -----------------------------------------------------------------------------
# 3. SHAP Values
# -----------------------------------------------------------------------------

def compute_shap_values(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    n_top: int = N_TOP_IMPORTANT_FEATURES,
) -> pd.DataFrame:
    """
    Compute SHAP (SHapley Additive exPlanations) values for the test set.
    SHAP values explain individual predictions by measuring each gene's
    contribution to pushing the prediction toward tumor (1) or normal (0).

    Two plots are generated:
        - SHAP summary dot plot  : Shows distribution of SHAP values per gene
        - SHAP bar plot          : Shows mean absolute SHAP value per gene

    Args:
        model : Trained RandomForestClassifier.
        X_test: Test feature matrix.
        n_top : Number of top features to display.

    Returns:
        DataFrame of genes with their mean absolute SHAP values, sorted descending.
    """
    logger.info("Computing SHAP values (TreeExplainer)...")
    logger.info("  This may take a few minutes for large feature sets...")

    # Load gene name mapping first
    gene_map = _get_gene_map()

    # Use newer shap Explanation API for compatibility with shap >= 0.40
    explainer_obj    = shap.TreeExplainer(model)
    shap_explanation = explainer_obj(X_test)

    # Extract SHAP values for positive class (tumor)
    # shap_explanation.values shape: (samples, features, classes)
    if shap_explanation.values.ndim == 3:
        shap_vals_positive = shap_explanation.values[:, :, 1]
        base_val_positive  = shap_explanation.base_values[:, 1] if shap_explanation.base_values.ndim == 2 else shap_explanation.base_values
    else:
        shap_vals_positive = shap_explanation.values
        base_val_positive  = shap_explanation.base_values

    logger.info(f"shap_vals_positive shape: {shap_vals_positive.shape}")

    # Build Explanation object for positive class only (for summary_plot)
    shap_exp_positive = shap.Explanation(
        values        = shap_vals_positive,
        base_values   = base_val_positive,
        data          = X_test.values,
        feature_names = apply_gene_names(X_test.columns.tolist(), gene_map),
    )

    # Mean absolute SHAP value per gene
    mean_abs_shap = np.abs(shap_vals_positive).mean(axis=0)
    df = pd.DataFrame({
        "gene"          : apply_gene_names(X_test.columns.tolist(), gene_map),
        "gene_id"       : X_test.columns.tolist(),
        "mean_abs_shap" : mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Save full SHAP importance table
    ensure_dir(METRICS_DIR)
    df.to_csv(SHAP_CSV_FILE, index=False)
    logger.info(f"SHAP importance saved to: {SHAP_CSV_FILE}")

    # --- SHAP Summary Dot Plot ---
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.sca(ax)
    shap.plots.beeswarm(
        shap_exp_positive,
        max_display=n_top,
        show=False,
    )
    ax.set_title(
        f"SHAP Summary Plot — Top {n_top} Genes (Tumor Class)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_FILE, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to: {SHAP_SUMMARY_FILE}")

    # --- SHAP Bar Plot (custom, using mean abs SHAP) ---
    top_df  = df.head(n_top).iloc[::-1]  # reverse for horizontal bar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_df["gene"],
        top_df["mean_abs_shap"],
        color="steelblue",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_ylabel("Gene",             fontsize=12)
    ax.set_title(
        f"SHAP Feature Importance — Top {n_top} Genes (Mean |SHAP|)",
        fontsize=13, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_FILE, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP bar plot saved to: {SHAP_BAR_FILE}")

    logger.info(f"Top 5 genes by SHAP: {df['gene'].head(5).tolist()}")
    return df


# -----------------------------------------------------------------------------
# Helper: Horizontal Bar Plot
# -----------------------------------------------------------------------------

def _plot_importance(
    df: pd.DataFrame,
    importance_col: str,
    std_col: str,
    title: str,
    xlabel: str,
    color: str,
    save_path: str,
) -> None:
    """
    Plot a horizontal bar chart of feature importances.

    Args:
        df             : DataFrame with gene and importance columns.
        importance_col : Column name for importance values.
        std_col        : Column name for standard deviation values.
        title          : Plot title.
        xlabel         : X-axis label.
        color          : Bar color.
        save_path      : File path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    sns.set_style(PLOT_STYLE)

    # Plot in descending order (most important at top)
    df_plot = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        df_plot["gene"],
        df_plot[importance_col],
        xerr=df_plot[std_col],
        color=color,
        alpha=0.85,
        capsize=3,
        edgecolor="white",
    )

    ax.set_xlabel(xlabel,  fontsize=12)
    ax.set_ylabel("Gene",  fontsize=12)
    ax.set_title(title,    fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()

    logger.info(f"Importance plot saved to: {save_path}")


# -----------------------------------------------------------------------------
# Full Explainability Pipeline
# -----------------------------------------------------------------------------

def explain(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_top: int = N_TOP_IMPORTANT_FEATURES,
) -> dict:
    """
    Full explainability pipeline for the final model.

    Runs all three importance methods and returns their results:
        1. MDI importance
        2. Permutation importance
        3. SHAP values

    Args:
        model : Trained final RandomForestClassifier.
        X_test: Test feature matrix.
        y_test: Test labels.
        n_top : Number of top features to display in plots.

    Returns:
        Dictionary with DataFrames for each importance method.
    """
    logger.info("=" * 60)
    logger.info("EXPLAINABILITY ANALYSIS STARTED")
    logger.info("=" * 60)

    feature_names = X_test.columns.tolist()

    # 1. MDI
    df_mdi = compute_mdi_importance(model, feature_names, n_top)

    # 2. Permutation
    df_perm = compute_permutation_importance(model, X_test, y_test, n_top)

    # 3. SHAP
    df_shap = compute_shap_values(model, X_test, n_top)

    # Summary: genes appearing in top N across all three methods
    top_mdi  = set(df_mdi.head(n_top)["gene"].tolist())
    top_perm = set(df_perm.head(n_top)["gene"].tolist())
    top_shap = set(df_shap.head(n_top)["gene"].tolist())

    consensus_genes = top_mdi & top_perm & top_shap
    logger.info(f"Consensus genes (top {n_top} in all 3 methods): {len(consensus_genes)}")
    logger.info(f"  {sorted(consensus_genes)}")

    # Save consensus gene list
    consensus_path = os.path.join(METRICS_DIR, "consensus_important_genes.txt")
    with open(consensus_path, "w") as f:
        f.write(f"# Genes appearing in top {n_top} across MDI, Permutation, and SHAP\n")
        for gene in sorted(consensus_genes):
            f.write(f"{gene}\n")
    logger.info(f"Consensus genes saved to: {consensus_path}")

    logger.info("EXPLAINABILITY ANALYSIS COMPLETED")
    logger.info("=" * 60)

    return {
        "mdi"        : df_mdi,
        "permutation": df_perm,
        "shap"       : df_shap,
        "consensus"  : consensus_genes,
    }
