# =============================================================================
# pipelines/run_data_loading.py
# -----------------------------------------------------------------------------
# Pipeline Step 1: Data Loading and Preprocessing
#
# - Loads raw RNA-seq TSV files from _raw_data/samples/
# - Assigns labels from GDC sample sheet
# - Filters low-expression genes
# - Applies log2(TPM + 1) transformation
# - Saves interim and processed feature matrices to data/
#
# Run this script once. Subsequent steps load from data/processed/.
#
# Usage:
#   python pipelines/run_data_loading.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from utils        import setup_logger, ensure_project_dirs, set_global_seed
from data_loader  import load_data
from preprocessor import preprocess

logger = setup_logger("run_data_loading", log_file=os.path.join(config.RESULTS_DIR, "data_loading.log"))


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: DATA LOADING & PREPROCESSING")
    logger.info("=" * 60)

    ensure_project_dirs(config)
    set_global_seed(config.SEED)

    # Load raw data
    X, y = load_data(samples_dir=config.SAMPLES_DIR, save_interim=True)

    # Preprocess
    X = preprocess(X, save_processed=True)

    # Save labels
    y.to_csv(os.path.join(config.PROCESSED_DIR, "labels.csv"), header=True)
    logger.info(f"Labels saved to: {config.PROCESSED_DIR}/labels.csv")

    logger.info("Step 1 complete. Outputs saved to data/")


if __name__ == "__main__":
    main()
