# =============================================================================
# pipelines/run_comparison.py
# -----------------------------------------------------------------------------
# Runs the full comparative study across all balancing strategies.
# Produces complete results for both CV and test set evaluation.
#
# Requires: run_data_loading.py to have been run first.
#
# Usage:
#   Run directly in VS Code or terminal:
#   python pipelines/run_comparison.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils        import setup_logger, ensure_project_dirs, set_global_seed, load_dataframe
from feature_selector import select_features
from comparator   import run_comparison
from balancer     import ALL_STRATEGIES

logger = setup_logger(
    "run_comparison",
    log_file=os.path.join(config.RESULTS_DIR, "comparison.log")
)

# Paths for saved split datasets (reuse from run_training if available)
TRAIN_X_PATH = os.path.join(config.PROCESSED_DIR, "X_train.csv")
TRAIN_Y_PATH = os.path.join(config.PROCESSED_DIR, "y_train.csv")
TEST_X_PATH  = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH  = os.path.join(config.PROCESSED_DIR, "y_test.csv")


def main():
    logger.info("=" * 60)
    logger.info("COMPARATIVE STUDY PIPELINE")
    logger.info("=" * 60)

    ensure_project_dirs(config)
    set_global_seed(config.SEED)

    # Load processed feature matrix and labels
    logger.info("Loading processed data...")
    X = load_dataframe(config.FEATURE_MATRIX_FINAL)
    y = pd.read_csv(
        os.path.join(config.PROCESSED_DIR, "labels.csv"), index_col=0
    ).squeeze()

    # Record gene counts for data summary
    n_genes_filtered = X.shape[1]

    # Load interim matrix to get pre-selection gene count
    try:
        X_interim    = load_dataframe(config.FEATURE_MATRIX_RAW)
        n_genes_raw  = X_interim.shape[1]
    except FileNotFoundError:
        n_genes_raw  = n_genes_filtered

    # Reuse existing train/test split if available, otherwise create new one
    if all(os.path.exists(p) for p in [TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TEST_Y_PATH]):
        logger.info("Loading existing train/test split...")
        X_train = load_dataframe(TRAIN_X_PATH)
        X_test  = load_dataframe(TEST_X_PATH)
        y_train = pd.read_csv(TRAIN_Y_PATH, index_col=0).squeeze()
        y_test  = pd.read_csv(TEST_Y_PATH,  index_col=0).squeeze()
    else:
        logger.info("Creating new train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = config.TEST_SIZE,
            random_state = config.RANDOM_STATE,
            stratify     = y,
        )
        # Feature selection
        X_train, X_test, _ = select_features(
            X_train=X_train, X_test=X_test, y_train=y_train,
            method=config.FEATURE_SELECTION_METHOD, n=config.N_TOP_FEATURES,
        )

    # Run full comparison
    run_comparison(
        X             = X,
        y             = y,
        X_train       = X_train,
        y_train       = y_train,
        X_test        = X_test,
        y_test        = y_test,
        n_genes_raw   = n_genes_raw,
        n_genes_filtered = n_genes_filtered,
        strategies    = ALL_STRATEGIES,
    )

    logger.info("Comparison pipeline complete.")


if __name__ == "__main__":
    main()
