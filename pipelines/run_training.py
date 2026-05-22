# =============================================================================
# pipelines/run_training.py
# -----------------------------------------------------------------------------
# Pipeline Step 2: Train/Test Split, Feature Selection, Model Training
#
# - Loads processed feature matrix from data/processed/
# - Splits into train/test (stratified, no leakage)
# - Selects features on training set only
# - Compares all balancing strategies via cross-validation
# - Trains final model with best strategy
# - Saves model, CV results, split datasets to disk
#
# Requires: run_data_loading.py to have been run first.
#
# Usage:
#   python pipelines/run_training.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils             import setup_logger, ensure_project_dirs, set_global_seed, load_dataframe, save_dataframe
from feature_selector  import select_features
from model             import run_model_pipeline

logger = setup_logger("run_training", log_file=os.path.join(config.RESULTS_DIR, "training.log"))

import logging
logging.getLogger("balancer").addHandler(
    logging.FileHandler(os.path.join(config.RESULTS_DIR, "training.log"))
)
logging.getLogger("model").addHandler(
    logging.FileHandler(os.path.join(config.RESULTS_DIR, "training.log"))
)


# Paths for saved split datasets
TRAIN_X_PATH = os.path.join(config.PROCESSED_DIR, "X_train.csv")
TRAIN_Y_PATH = os.path.join(config.PROCESSED_DIR, "y_train.csv")
TEST_X_PATH  = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH  = os.path.join(config.PROCESSED_DIR, "y_test.csv")


def main():
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE SELECTION & MODEL TRAINING")
    logger.info("=" * 60)

    ensure_project_dirs(config)
    set_global_seed(config.SEED)

    # Load processed data
    logger.info("Loading processed feature matrix...")
    X = load_dataframe(config.FEATURE_MATRIX_FINAL)
    y = pd.read_csv(os.path.join(config.PROCESSED_DIR, "labels.csv"), index_col=0).squeeze()

    # Train/Test Split
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.TEST_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y,
    )
    logger.info(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Feature Selection (train only)
    X_train, X_test, _ = select_features(
        X_train = X_train,
        X_test  = X_test,
        y_train = y_train,
        method  = config.FEATURE_SELECTION_METHOD,
        n       = config.N_TOP_FEATURES,
    )

    # Save splits for evaluation step
    save_dataframe(X_train, TRAIN_X_PATH)
    save_dataframe(X_test,  TEST_X_PATH)
    y_train.to_csv(TRAIN_Y_PATH, header=True)
    y_test.to_csv(TEST_Y_PATH,   header=True)
    logger.info("Train/test splits saved to data/processed/")

    # Model training
    run_model_pipeline(X_train=X_train, y_train=y_train)

    logger.info("Step 2 complete. Model saved to results/")


if __name__ == "__main__":
    main()
