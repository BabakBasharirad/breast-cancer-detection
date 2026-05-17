# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Central configuration file for the Breast Cancer Detection pipeline.
# All paths, constants, and hyperparameters are defined here.
# Modify this file to adapt the pipeline to your environment.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# Project Root
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Raw Data Paths
# -----------------------------------------------------------------------------
RAW_DATA_DIR        = os.path.join(PROJECT_ROOT, "_raw_data")
SAMPLES_DIR         = os.path.join(RAW_DATA_DIR, "samples")
SAMPLE_SHEET_PATH   = os.path.join(RAW_DATA_DIR, "gdc_sample_sheet.tsv")
MANIFEST_PATH       = os.path.join(RAW_DATA_DIR, "gdc_manifest.txt")

# -----------------------------------------------------------------------------
# Data Paths (Interim & Processed)
# -----------------------------------------------------------------------------
DATA_DIR            = os.path.join(PROJECT_ROOT, "data")
INTERIM_DIR         = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR       = os.path.join(DATA_DIR, "processed")

FEATURE_MATRIX_RAW  = os.path.join(INTERIM_DIR, "feature_matrix_raw.csv")
FEATURE_MATRIX_FINAL= os.path.join(PROCESSED_DIR, "feature_matrix_final.csv")

# -----------------------------------------------------------------------------
# Results Paths
# -----------------------------------------------------------------------------
RESULTS_DIR         = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR         = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR         = os.path.join(RESULTS_DIR, "metrics")

# -----------------------------------------------------------------------------
# GDC / Data Loading Settings
# -----------------------------------------------------------------------------
# Column to extract from each RNA-seq TSV file
TPM_COLUMN          = "tpm_unstranded"

# Column names in the GDC sample sheet
SAMPLE_SHEET_FILE_ID_COL   = "File ID"
SAMPLE_SHEET_SAMPLE_TYPE   = "Tissue Type"

# Sample type labels as they appear in the GDC sample sheet
TUMOR_LABEL         = "Tumor"
NORMAL_LABEL        = "Normal"

# Binary label mapping
LABEL_MAP = {
    TUMOR_LABEL:  1,
    NORMAL_LABEL: 0
}

# Number of header/metadata rows to skip in each GDC TSV file
GDC_TSV_SKIP_ROWS   = 1  # First row after header is metadata

# Gene biotype to keep (set to None to keep all)
GENE_TYPE_FILTER    = "protein_coding"

# -----------------------------------------------------------------------------
# Preprocessing Settings
# -----------------------------------------------------------------------------
# Log transformation: log2(TPM + 1)
LOG_TRANSFORM       = True
LOG_BASE            = 2
LOG_PSEUDOCOUNT     = 1

# Minimum TPM threshold: remove genes with low expression across samples
MIN_TPM_THRESHOLD   = 0.1          # Genes below this in most samples are dropped
MIN_SAMPLE_FRACTION = 0.20         # Gene must be expressed in at least 20% of samples

# -----------------------------------------------------------------------------
# Feature Selection Settings
# -----------------------------------------------------------------------------
FEATURE_SELECTION_METHOD    = "variance"    # Options: "variance", "mutual_info", "all"
N_TOP_FEATURES              = 2000          # Number of top features to retain

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
RANDOM_STATE        = 42
TEST_SIZE           = 0.20          # 80/20 train-test split
CV_FOLDS            = 5             # Cross-validation folds

# Random Forest Hyperparameters
RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":        None,
    "min_samples_split":2,
    "min_samples_leaf": 1,
    "max_features":     "sqrt",
    "class_weight":     "balanced",  # Handles class imbalance (tumor >> normal)
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1           # Use all available CPU cores
}

# -----------------------------------------------------------------------------
# Evaluation Settings
# -----------------------------------------------------------------------------
METRICS_OUTPUT_FILE = os.path.join(METRICS_DIR, "evaluation_metrics.json")
ROC_CURVE_FILE      = os.path.join(FIGURES_DIR, "roc_curve.png")
CONFUSION_MATRIX_FILE = os.path.join(FIGURES_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_FILE = os.path.join(FIGURES_DIR, "feature_importance.png")

# Number of top important features to display in the feature importance plot
N_TOP_IMPORTANT_FEATURES = 20

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = RANDOM_STATE
