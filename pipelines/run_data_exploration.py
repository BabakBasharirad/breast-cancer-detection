# =============================================================================
# pipelines/run_data_exploration.py
# -----------------------------------------------------------------------------
# Generates data exploration and visualization plots for the paper.
#
# Plots produced:
#   1. Class distribution bar chart
#   2. Gene expression distribution histogram (tumor vs normal)
#   3. PCA scatter plot (2D) of selected features
#   4. Heatmap of top 50 most variable genes
#   5. Variance distribution before and after feature selection
#
# Requires: run_data_loading.py and run_training.py to have been run first.
#
# Usage:
#   Run directly in VS Code or terminal:
#   python pipelines/run_data_exploration.py
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from utils import setup_logger, ensure_dir, load_dataframe

logger = setup_logger(
    "run_data_exploration",
    log_file=os.path.join(config.RESULTS_DIR, "data_exploration.log")
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EXPLORATION_DIR = os.path.join(config.FIGURES_DIR, "exploration")
TRAIN_X_PATH    = os.path.join(config.PROCESSED_DIR, "X_train.csv")
TRAIN_Y_PATH    = os.path.join(config.PROCESSED_DIR, "y_train.csv")
TEST_X_PATH     = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH     = os.path.join(config.PROCESSED_DIR, "y_test.csv")

PLOT_DPI    = 300
PLOT_FORMAT = "png"
PLOT_STYLE  = "whitegrid"

TUMOR_COLOR  = "#E74C3C"
NORMAL_COLOR = "#2980B9"


# -----------------------------------------------------------------------------
# Plot 1: Class Distribution
# -----------------------------------------------------------------------------

def plot_class_distribution(y: pd.Series) -> None:
    """Bar chart showing tumor vs normal sample counts."""
    ensure_dir(EXPLORATION_DIR)
    sns.set_style(PLOT_STYLE)

    counts     = y.value_counts().sort_index()
    labels     = ["Normal (0)", "Tumor (1)"]
    colors     = [NORMAL_COLOR, TUMOR_COLOR]
    values     = [counts.get(0, 0), counts.get(1, 0)]
    total      = sum(values)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution — TCGA-BRCA Dataset", fontsize=13, fontweight="bold")
    ax.set_ylim([0, max(values) * 1.2])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = os.path.join(EXPLORATION_DIR, "class_distribution.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Class distribution plot saved to: {path}")


# -----------------------------------------------------------------------------
# Plot 2: Gene Expression Distribution
# -----------------------------------------------------------------------------

def plot_expression_distribution(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Histogram of log2(TPM+1) values for tumor and normal samples.
    Samples a subset of genes to keep the plot readable.
    """
    ensure_dir(EXPLORATION_DIR)
    sns.set_style(PLOT_STYLE)

    # Sample up to 500 genes for readability
    n_genes   = min(500, X.shape[1])
    gene_sample = np.random.RandomState(config.RANDOM_STATE).choice(X.columns, n_genes, replace=False)

    tumor_vals  = X.loc[y == 1, gene_sample].values.flatten()
    normal_vals = X.loc[y == 0, gene_sample].values.flatten()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(tumor_vals,  bins=80, alpha=0.6, color=TUMOR_COLOR,  label="Tumor",  density=True)
    ax.hist(normal_vals, bins=80, alpha=0.6, color=NORMAL_COLOR, label="Normal", density=True)

    ax.set_xlabel(r"$\log_2(\mathrm{TPM} + 1)$", fontsize=12)
    ax.set_ylabel("Density",                       fontsize=12)
    ax.set_title(
        "Gene Expression Distribution — Tumor vs Normal",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(EXPLORATION_DIR, "expression_distribution.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Expression distribution plot saved to: {path}")


# -----------------------------------------------------------------------------
# Plot 3: PCA Scatter Plot
# -----------------------------------------------------------------------------

def plot_pca(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """2D PCA scatter plot of selected features, colored by class."""
    ensure_dir(EXPLORATION_DIR)
    sns.set_style(PLOT_STYLE)

    logger.info("Computing PCA (2 components)...")
    pca        = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca      = pca.fit_transform(X)
    var_exp    = pca.explained_variance_ratio_ * 100

    df_pca     = pd.DataFrame({
        "PC1"   : X_pca[:, 0],
        "PC2"   : X_pca[:, 1],
        "Class" : y.map({1: "Tumor", 0: "Normal"}).values,
    })

    fig, ax = plt.subplots(figsize=(9, 7))

    for label, color in [("Tumor", TUMOR_COLOR), ("Normal", NORMAL_COLOR)]:
        subset = df_pca[df_pca["Class"] == label]
        ax.scatter(
            subset["PC1"], subset["PC2"],
            c=color, label=label, alpha=0.6, s=20, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=12)
    ax.set_title(
        "PCA Scatter Plot — Tumor vs Normal (2,000 Selected Genes)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11, markerscale=2)

    plt.tight_layout()
    path = os.path.join(EXPLORATION_DIR, "pca_scatter.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"PCA scatter plot saved to: {path}")


# -----------------------------------------------------------------------------
# Plot 4: Heatmap of Top 50 Most Variable Genes
# -----------------------------------------------------------------------------

def plot_expression_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    n_genes: int = 50,
) -> None:
    """
    Heatmap of top N most variable genes across all samples.
    Samples sorted by class for clear visual separation.
    """
    ensure_dir(EXPLORATION_DIR)

    # Select top N most variable genes
    top_genes  = X.var(axis=0).nlargest(n_genes).index
    X_top      = X[top_genes]

    # Sort samples by class (normal first, tumor second)
    sort_idx   = y.sort_values().index
    X_sorted   = X_top.loc[sort_idx]
    y_sorted   = y.loc[sort_idx]

    # Create class color bar
    class_colors = y_sorted.map({1: TUMOR_COLOR, 0: NORMAL_COLOR})

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        X_sorted.T,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=True,
        ax=ax,
        cbar_kws={"label": r"$\log_2(\mathrm{TPM}+1)$", "shrink": 0.6},
    )

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Gene",    fontsize=11)
    ax.set_title(
        f"Expression Heatmap — Top {n_genes} Most Variable Genes",
        fontsize=13, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=7)

    # Add class color bar on top
    ax2 = ax.inset_axes([0, 1.01, 1, 0.03])
    numeric_colors = y_sorted.map({1: 1, 0: 0}).values.reshape(1, -1)
    ax2.imshow(
        numeric_colors,
        aspect="auto",
        interpolation="none",
        cmap=plt.cm.get_cmap('bwr', 2),
        vmin=0, vmax=1,
    )
    ax2.set_axis_off()

    # Add legend for class colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TUMOR_COLOR,  label="Tumor"),
        Patch(facecolor=NORMAL_COLOR, label="Normal"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.12, 1.15),
        fontsize=9,
    )

    plt.tight_layout()
    path = os.path.join(EXPLORATION_DIR, "expression_heatmap.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Expression heatmap saved to: {path}")


# -----------------------------------------------------------------------------
# Plot 5: Variance Distribution
# -----------------------------------------------------------------------------

def plot_variance_distribution(
    X_full: pd.DataFrame,
    X_selected: pd.DataFrame,
) -> None:
    """
    Distribution of gene variances before and after feature selection.
    Shows the effect of the variance-based filter.
    """
    ensure_dir(EXPLORATION_DIR)
    sns.set_style(PLOT_STYLE)

    var_full     = X_full.var(axis=0)
    var_selected = X_selected.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before selection
    axes[0].hist(var_full, bins=80, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Variance", fontsize=11)
    axes[0].set_ylabel("Number of Genes", fontsize=11)
    axes[0].set_title(f"All Genes (n={X_full.shape[1]:,})", fontsize=12, fontweight="bold")
    axes[0].set_yscale("log")

    # After selection
    axes[1].hist(var_selected, bins=80, color=TUMOR_COLOR, alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Variance", fontsize=11)
    axes[1].set_ylabel("Number of Genes", fontsize=11)
    axes[1].set_title(f"Selected Genes (n={X_selected.shape[1]:,})", fontsize=12, fontweight="bold")
    axes[1].set_yscale("log")

    fig.suptitle(
        "Gene Variance Distribution — Before and After Feature Selection",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(EXPLORATION_DIR, "variance_distribution.png")
    plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches="tight")
    plt.close()
    logger.info(f"Variance distribution plot saved to: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("DATA EXPLORATION STARTED")
    logger.info("=" * 60)

    ensure_dir(EXPLORATION_DIR)

    # Load processed full matrix (for variance distribution)
    logger.info("Loading processed data...")
    X_full = load_dataframe(config.FEATURE_MATRIX_FINAL)
    y_full = pd.read_csv(
        os.path.join(config.PROCESSED_DIR, "labels.csv"), index_col=0
    ).squeeze()

    # Load train/test splits (selected features)
    X_train = load_dataframe(TRAIN_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH, index_col=0).squeeze()
    X_test  = load_dataframe(TEST_X_PATH)
    y_test  = pd.read_csv(TEST_Y_PATH,  index_col=0).squeeze()

    # Combine train and test for full dataset plots
    X_selected = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, y_test], axis=0)

    # Plot 1: Class distribution
    logger.info("Plot 1: Class distribution...")
    plot_class_distribution(y_full)

    # Plot 2: Expression distribution
    logger.info("Plot 2: Expression distribution...")
    plot_expression_distribution(X_selected, y_combined)

    # Plot 3: PCA scatter
    logger.info("Plot 3: PCA scatter plot...")
    plot_pca(X_selected, y_combined)

    # Plot 4: Expression heatmap
    logger.info("Plot 4: Expression heatmap...")
    plot_expression_heatmap(X_selected, y_combined)

    # Plot 5: Variance distribution
    logger.info("Plot 5: Variance distribution...")
    plot_variance_distribution(X_full, X_selected)

    logger.info("=" * 60)
    logger.info("DATA EXPLORATION COMPLETED")
    logger.info(f"All plots saved to: {EXPLORATION_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
