# =============================================================================
# feature_selector.py
# -----------------------------------------------------------------------------
# Performs feature selection on the training set only to prevent data leakage.
# The selected features are then applied to transform the test set.
#
# Steps:
#   1. Variance thresholding  — remove near-zero variance genes
#   2. Top-N by variance      — keep the most variable genes
#   3. (Optional) Mutual Info — supervised, label-aware selection
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from config import (
    FEATURE_SELECTION_METHOD,
    N_TOP_FEATURES,
    METRICS_DIR,
)
from utils import (
    setup_logger,
    ensure_dir,
    log_dataframe_info,
)

logger = setup_logger(__name__)

# Path to save selected gene list
SELECTED_GENES_PATH = os.path.join(METRICS_DIR, "selected_genes.json")


# -----------------------------------------------------------------------------
# Step 1: Variance Thresholding
# -----------------------------------------------------------------------------

def remove_zero_variance_genes(X_train: pd.DataFrame) -> list:
    """
    Identify genes with non-zero variance using VarianceThreshold.
    Fitted on training data only.

    Args:
        X_train : Training feature matrix (samples × genes).

    Returns:
        List of gene names with non-zero variance.
    """
    logger.info("Removing near-zero variance genes...")

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)

    genes_kept = X_train.columns[selector.get_support()].tolist()
    n_removed  = X_train.shape[1] - len(genes_kept)

    logger.info(f"  Genes before variance filter : {X_train.shape[1]:,}")
    logger.info(f"  Genes after variance filter  : {len(genes_kept):,}")
    logger.info(f"  Genes removed                : {n_removed:,}")

    return genes_kept


# -----------------------------------------------------------------------------
# Step 2: Top-N Genes by Variance
# -----------------------------------------------------------------------------

def select_top_n_by_variance(
    X_train: pd.DataFrame,
    n: int = N_TOP_FEATURES,
) -> list:
    """
    Select the top N most variable genes based on variance across
    training samples. High variance genes are more likely to be
    discriminative between tumor and normal samples.

    Args:
        X_train : Training feature matrix (samples × genes).
        n       : Number of top genes to retain.

    Returns:
        List of top N gene names ranked by variance.
    """
    logger.info(f"Selecting top {n:,} genes by variance...")

    variances  = X_train.var(axis=0)
    top_genes  = variances.nlargest(n).index.tolist()

    logger.info(f"  Top {n:,} genes selected by variance.")

    return top_genes


# ----------------------------------------------------------------------------------------------
# Additional Optional Step: Mutual Information (This method is not applied in this experiment.)
# ----------------------------------------------------------------------------------------------

def select_top_n_by_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n: int = N_TOP_FEATURES,
) -> list:
    """
    Select the top N genes based on mutual information with the class label.
    This is a supervised method — it measures how much each gene's expression
    depends on the tumor/normal label.

    Note: Slower than variance-based selection but more discriminative.

    Args:
        X_train : Training feature matrix (samples × genes).
        y_train : Training labels (0=Normal, 1=Tumor).
        n       : Number of top genes to retain.

    Returns:
        List of top N gene names ranked by mutual information score.
    """
    logger.info(f"Selecting top {n:,} genes by mutual information...")

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=False,
        random_state=42,
    )

    mi_series = pd.Series(mi_scores, index=X_train.columns)
    top_genes = mi_series.nlargest(n).index.tolist()

    logger.info(f"  Top {n:,} genes selected by mutual information.")

    return top_genes


# -----------------------------------------------------------------------------
# Main Feature Selection Pipeline
# -----------------------------------------------------------------------------

def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    method: str = FEATURE_SELECTION_METHOD,
    n: int = N_TOP_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Full feature selection pipeline. Fitted on training data only.
    The same selected genes are used to transform the test set.

    Pipeline:
        1. Remove zero-variance genes (always applied)
        2. Select top N genes by variance OR mutual information
           based on the configured method

    Args:
        X_train : Training feature matrix (samples × genes).
        X_test  : Test feature matrix (samples × genes).
        y_train : Training labels (0=Normal, 1=Tumor).
        method  : Selection method — 'variance', 'mutual_info', or 'all'
        n       : Number of top features to retain.

    Returns:
        Tuple of:
            - X_train_selected : Reduced training matrix
            - X_test_selected  : Reduced test matrix (same genes)
            - selected_genes   : List of selected gene names
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION STARTED")
    logger.info(f"Method: {method} | Top N: {n:,}")
    logger.info("=" * 60)

    log_dataframe_info(X_train, "X_train (input)", logger)
    log_dataframe_info(X_test,  "X_test  (input)", logger)

    # Step 1: Remove zero-variance genes (fitted on train only)
    nonzero_genes = remove_zero_variance_genes(X_train)
    X_train = X_train[nonzero_genes]
    X_test  = X_test[nonzero_genes]

    # Step 2: Select top N genes
    if method == "variance":
        selected_genes = select_top_n_by_variance(X_train, n)

    elif method == "mutual_info":
        selected_genes = select_top_n_by_mutual_info(X_train, y_train, n)

    elif method == "all":
        # Run both and take the union
        logger.info("Running both variance and mutual info selection (union)...")
        var_genes = select_top_n_by_variance(X_train, n)
        mi_genes  = select_top_n_by_mutual_info(X_train, y_train, n)
        selected_genes = list(set(var_genes) | set(mi_genes))
        logger.info(f"  Union of both methods: {len(selected_genes):,} genes")

    else:
        raise ValueError(
            f"Unknown feature selection method: '{method}'. "
            f"Choose from: 'variance', 'mutual_info', 'all'."
        )

    # Apply selection to both train and test
    X_train_selected = X_train[selected_genes]
    X_test_selected  = X_test[selected_genes]

    log_dataframe_info(X_train_selected, "X_train (selected)", logger)
    log_dataframe_info(X_test_selected,  "X_test  (selected)", logger)

    # Save selected gene list
    _save_selected_genes(selected_genes)

    logger.info("FEATURE SELECTION COMPLETED")
    logger.info("=" * 60)

    return X_train_selected, X_test_selected, selected_genes


# -----------------------------------------------------------------------------
# Helper: Save Selected Genes
# -----------------------------------------------------------------------------

def _save_selected_genes(genes: list) -> None:
    """
    Save the list of selected gene IDs to a JSON file for reproducibility
    and reporting in the paper.

    Args:
        genes : List of selected gene names/IDs.
    """
    ensure_dir(METRICS_DIR)
    with open(SELECTED_GENES_PATH, "w") as f:
        json.dump({"n_selected": len(genes), "genes": genes}, f, indent=4)
    logger.info(f"  Selected genes saved to: {SELECTED_GENES_PATH}")


def load_selected_genes(path: str = SELECTED_GENES_PATH) -> list:
    """
    Load the previously saved list of selected genes.

    Args:
        path : Path to the saved JSON file.

    Returns:
        List of selected gene names/IDs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected genes file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data["genes"]
