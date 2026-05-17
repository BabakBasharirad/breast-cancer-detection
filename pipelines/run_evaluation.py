# =============================================================================
# pipelines/run_evaluation.py
# -----------------------------------------------------------------------------
# Pipeline Step 3: Model Evaluation on Test Set
#
# - Loads final model and test set from disk
# - Computes classification metrics
# - Generates confusion matrix, ROC curve, strategy comparison plots
# - Saves all metrics and figures to results/
#
# Requires: run_training.py to have been run first.
#
# Usage:
#   python pipelines/run_evaluation.py
# =============================================================================

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

import config
from utils      import setup_logger, ensure_project_dirs, load_dataframe
from model      import load_model
from evaluator  import evaluate

logger = setup_logger("run_evaluation", log_file=os.path.join(config.RESULTS_DIR, "evaluation.log"))

TRAIN_X_PATH = os.path.join(config.PROCESSED_DIR, "X_train.csv")
TRAIN_Y_PATH = os.path.join(config.PROCESSED_DIR, "y_train.csv")
TEST_X_PATH  = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH  = os.path.join(config.PROCESSED_DIR, "y_test.csv")


def main():
    logger.info("=" * 60)
    logger.info("STEP 3: EVALUATION")
    logger.info("=" * 60)

    ensure_project_dirs(config)

    # Load test set
    logger.info("Loading test set...")
    X_test  = load_dataframe(TEST_X_PATH)
    y_test  = pd.read_csv(TEST_Y_PATH, index_col=0).squeeze()

    # Load final model
    model = load_model()

    # Load CV results and best strategy
    best_strategy_path = os.path.join(config.METRICS_DIR, "best_strategy.json")
    cv_results_path    = os.path.join(config.METRICS_DIR, "cv_results_all_strategies.csv")

    with open(best_strategy_path, "r") as f:
        best_info = json.load(f)
    best_strategy = best_info["best_strategy"]

    cv_results = pd.read_csv(cv_results_path)

    # Evaluate
    evaluate(
        model         = model,
        X_test        = X_test,
        y_test        = y_test,
        cv_results    = cv_results,
        best_strategy = best_strategy,
    )

    logger.info("Step 3 complete. Results saved to results/")


if __name__ == "__main__":
    main()
