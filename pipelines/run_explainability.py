# =============================================================================
# pipelines/run_explainability.py
# -----------------------------------------------------------------------------
# Pipeline Step 4: Model Explainability
#
# - Loads final model and test set from disk
# - Computes MDI, permutation importance, and SHAP values
# - Saves all importance figures and CSVs to results/
#
# Requires: run_training.py to have been run first.
#
# Usage:
#   python pipelines/run_explainability.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

import config
from utils      import setup_logger, ensure_project_dirs, load_dataframe
from model      import load_model
from explainer  import explain

logger = setup_logger("run_explainability", log_file=os.path.join(config.RESULTS_DIR, "explainability.log"))

TEST_X_PATH = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH = os.path.join(config.PROCESSED_DIR, "y_test.csv")


def main():
    logger.info("=" * 60)
    logger.info("STEP 4: EXPLAINABILITY")
    logger.info("=" * 60)

    ensure_project_dirs(config)

    # Load test set
    logger.info("Loading test set...")
    X_test = load_dataframe(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH, index_col=0).squeeze()

    # Load final model
    model = load_model()

    # Run explainability
    explain(
        model  = model,
        X_test = X_test,
        y_test = y_test,
        n_top  = config.N_TOP_IMPORTANT_FEATURES,
    )

    logger.info("Step 4 complete. Figures and CSVs saved to results/")


if __name__ == "__main__":
    main()
