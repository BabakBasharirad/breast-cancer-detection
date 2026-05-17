# =============================================================================
# pipelines/run_all.py
# -----------------------------------------------------------------------------
# Full pipeline runner — executes all steps end-to-end.
# Use this for complete reproduction of all results.
#
# Steps:
#   1. Data Loading & Preprocessing
#   2. Feature Selection & Model Training
#   3. Evaluation
#   4. Explainability
#
# Usage:
#   python pipelines/run_all.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from utils import setup_logger

logger = setup_logger("run_all", log_file=os.path.join(config.RESULTS_DIR, "run_all.log"))


def main():
    logger.info("=" * 60)
    logger.info("BREAST CANCER DETECTION — FULL PIPELINE")
    logger.info("TCGA-BRCA | RNA-Seq | Random Forest")
    logger.info("=" * 60)

    # Step 1
    logger.info("\nRunning Step 1: Data Loading & Preprocessing...")
    from pipelines.run_data_loading import main as run_data_loading
    run_data_loading()

    # Step 2
    logger.info("\nRunning Step 2: Feature Selection & Model Training...")
    from pipelines.run_training import main as run_training
    run_training()

    # Step 3
    logger.info("\nRunning Step 3: Evaluation...")
    from pipelines.run_evaluation import main as run_evaluation
    run_evaluation()

    # Step 4
    logger.info("\nRunning Step 4: Explainability...")
    from pipelines.run_explainability import main as run_explainability
    run_explainability()

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"  Metrics : {config.METRICS_DIR}")
    logger.info(f"  Figures : {config.FIGURES_DIR}")
    logger.info(f"  Model   : {os.path.join(config.RESULTS_DIR, 'final_model.pkl')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
