# =============================================================================
# preprocessor.py
# -----------------------------------------------------------------------------
# Handles all preprocessing steps applied to the raw feature matrix before
# feature selection and model training. Steps include:
#   1. Removing low-expression genes
#   2. Log2(TPM + 1) transformation
#   3. Saving the final processed matrix
# =============================================================================

import numpy as np
import pandas as pd

from config import (
    FEATURE_MATRIX_FINAL,
    LOG_TRANSFORM,
    LOG_BASE,
    LOG_PSEUDOCOUNT,
    MIN_TPM_THRESHOLD,
    MIN_SAMPLE_FRACTION,
)
from utils import (
    setup_logger,
    save_dataframe,
    log_dataframe_info,
)

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Step 1: Remove Low-Expression Genes
# -----------------------------------------------------------------------------

def filter_low_expression_genes(
    X: pd.DataFrame,
    min_tpm: float = MIN_TPM_THRESHOLD,
    min_fraction: float = MIN_SAMPLE_FRACTION,
) -> pd.DataFrame:
    """
    Remove genes that are not sufficiently expressed across samples.
    A gene is retained only if its TPM value exceeds `min_tpm` in at
    least `min_fraction` of all samples.

    Args:
        X            : Raw feature matrix (samples × genes).
        min_tpm      : Minimum TPM threshold to consider a gene expressed.
        min_fraction : Minimum fraction of samples where the gene must be expressed.

    Returns:
        Filtered feature matrix with low-expression genes removed.
    """
    logger.info("Filtering low-expression genes...")
    logger.info(f"  Minimum TPM threshold : {min_tpm}")
    logger.info(f"  Minimum sample fraction: {min_fraction * 100:.0f}%")

    n_genes_before = X.shape[1]

    # Boolean mask: is each gene expressed (TPM > threshold) per sample?
    expressed      = (X > min_tpm)

    # Fraction of samples where each gene is expressed
    expressed_frac = expressed.mean(axis=0)

    # Keep genes expressed in at least min_fraction of samples
    genes_to_keep  = expressed_frac[expressed_frac >= min_fraction].index
    X_filtered     = X[genes_to_keep]

    n_genes_after  = X_filtered.shape[1]
    n_removed      = n_genes_before - n_genes_after

    logger.info(f"  Genes before filtering : {n_genes_before:,}")
    logger.info(f"  Genes after filtering  : {n_genes_after:,}")
    logger.info(f"  Genes removed          : {n_removed:,}")

    return X_filtered


# -----------------------------------------------------------------------------
# Step 2: Log Transformation
# -----------------------------------------------------------------------------

def log_transform(
    X: pd.DataFrame,
    base: int = LOG_BASE,
    pseudocount: float = LOG_PSEUDOCOUNT,
) -> pd.DataFrame:
    """
    Apply log transformation to TPM values to reduce skewness.
    Formula: log2(TPM + 1)

    The pseudocount prevents log(0) errors for zero-expressed genes.

    Args:
        X           : Feature matrix (samples × genes).
        base        : Logarithm base (default: 2).
        pseudocount : Value added before log transformation (default: 1).

    Returns:
        Log-transformed feature matrix.
    """
    logger.info(f"Applying log{base}(TPM + {pseudocount}) transformation...")

    if base == 2:
        X_log = np.log2(X + pseudocount)
    elif base == 10:
        X_log = np.log10(X + pseudocount)
    else:
        X_log = np.log(X + pseudocount) / np.log(base)

    # Preserve DataFrame structure
    X_log = pd.DataFrame(X_log, index=X.index, columns=X.columns)

    logger.info("  Log transformation applied.")
    logger.info(f"  Value range after transform — min: {X_log.values.min():.4f}, max: {X_log.values.max():.4f}")

    return X_log


# -----------------------------------------------------------------------------
# Main Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess(
    X: pd.DataFrame,
    save_processed: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline applied to the raw feature matrix:
        1. Filter low-expression genes
        2. Apply log2(TPM + 1) transformation
        3. Save the processed matrix to disk

    Args:
        X              : Raw feature matrix (samples × genes).
        save_processed : Whether to save the processed matrix to disk.

    Returns:
        Preprocessed feature matrix ready for feature selection.
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING STARTED")
    logger.info("=" * 60)

    log_dataframe_info(X, "Input matrix", logger)

    # Step 1: Filter low-expression genes
    X = filter_low_expression_genes(X)

    # Step 2: Log transformation
    if LOG_TRANSFORM:
        X = log_transform(X)
    else:
        logger.info("Log transformation skipped (LOG_TRANSFORM=False in config).")

    log_dataframe_info(X, "Processed matrix", logger)

    # Step 3: Save to disk
    if save_processed:
        logger.info(f"Saving processed feature matrix to: {FEATURE_MATRIX_FINAL}")
        save_dataframe(X, FEATURE_MATRIX_FINAL, index=True)

    logger.info("PREPROCESSING COMPLETED")
    logger.info("=" * 60)

    return X
