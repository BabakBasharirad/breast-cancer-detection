# =============================================================================
# data_loader.py
# -----------------------------------------------------------------------------
# Handles loading and merging of GDC RNA-seq TSV files into a single feature
# matrix, and assigns binary labels using the GDC sample sheet.
# =============================================================================

import os
import pandas as pd
from tqdm import tqdm

from config import (
    SAMPLES_DIR,
    SAMPLE_SHEET_PATH,
    FEATURE_MATRIX_RAW,
    METRICS_DIR,
    TPM_COLUMN,
    SAMPLE_SHEET_FILE_ID_COL,
    SAMPLE_SHEET_SAMPLE_TYPE,
    TUMOR_LABEL,
    NORMAL_LABEL,
    LABEL_MAP,
    GDC_TSV_SKIP_ROWS,
    GENE_TYPE_FILTER,
)
from utils import (
    setup_logger,
    save_dataframe,
    log_dataframe_info,
    check_class_balance,
    ensure_dir,
)

logger = setup_logger(__name__)

# Path to save gene_id -> gene_name mapping
GENE_NAME_MAP_PATH = os.path.join(METRICS_DIR, "gene_name_mapping.json")


# -----------------------------------------------------------------------------
# Sample Sheet
# -----------------------------------------------------------------------------

def load_sample_sheet(path: str = SAMPLE_SHEET_PATH) -> pd.DataFrame:
    """
    Load the GDC sample sheet and extract File ID and Sample Type columns.
    Retains only tumor and normal samples and maps them to binary labels.

    Args:
        path: Path to the GDC sample sheet TSV file.

    Returns:
        DataFrame with columns: ['file_id', 'sample_type', 'label']
    """
    logger.info(f"Loading sample sheet from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample sheet not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # Keep only relevant columns
    df = df[[SAMPLE_SHEET_FILE_ID_COL, SAMPLE_SHEET_SAMPLE_TYPE]].copy()
    df.columns = ["file_id", "sample_type"]

    # Keep only tumor and normal samples
    valid_types = [TUMOR_LABEL, NORMAL_LABEL]
    df = df[df["sample_type"].isin(valid_types)].copy()

    # Map to binary labels
    df["label"] = df["sample_type"].map(LABEL_MAP)

    logger.info(f"Sample sheet loaded — {len(df)} samples retained.")
    logger.info(f"  Tumor  (1): {(df['label'] == 1).sum()}")
    logger.info(f"  Normal (0): {(df['label'] == 0).sum()}")

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Single TSV File Loader
# -----------------------------------------------------------------------------

def load_single_tsv(file_id: str, samples_dir: str = SAMPLES_DIR) -> pd.Series | None:
    """
    Load TPM values from a single GDC RNA-seq TSV file.

    GDC TSV structure:
        - Row 0     : Column headers
        - Row 1     : Metadata row (skipped)
        - Row 2+    : Gene data

    Args:
        file_id     : GDC file UUID (subfolder name under samples_dir).
        samples_dir : Root directory containing sample subfolders.

    Returns:
        A pandas Series indexed by gene_id with TPM values,
        or None if the file cannot be loaded.
    """
    subfolder = os.path.join(samples_dir, file_id)

    if not os.path.exists(subfolder):
        logger.warning(f"Subfolder not found for file_id: {file_id}")
        return None

    # Find the TSV file inside the subfolder
    tsv_files = [f for f in os.listdir(subfolder) if f.endswith(".tsv")]

    if len(tsv_files) == 0:
        logger.warning(f"No TSV file found in: {subfolder}")
        return None

    if len(tsv_files) > 1:
        logger.warning(f"Multiple TSV files found in: {subfolder} — using first.")

    tsv_path = os.path.join(subfolder, tsv_files[0])

    try:
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            skiprows=GDC_TSV_SKIP_ROWS,   # Skip metadata row
            comment="#",
        )

        # Filter to protein-coding genes only (if configured)
        if GENE_TYPE_FILTER and "gene_type" in df.columns:
            df = df[df["gene_type"] == GENE_TYPE_FILTER]

        # Set gene_id as index and extract TPM column
        df = df.set_index("gene_id")

        # Save gene_id -> gene_name mapping AFTER setting index
        if "gene_name" in df.columns:
            gene_map = df["gene_name"].to_dict()
        else:
            gene_map = {}

        if TPM_COLUMN not in df.columns:
            logger.warning(f"Column '{TPM_COLUMN}' not found in: {tsv_path}")
            return None

        tpm_series = df[TPM_COLUMN].astype(float)
        tpm_series.name = file_id

        return tpm_series, gene_map

    except Exception as e:
        logger.error(f"Failed to load {tsv_path}: {e}")
        return None


# -----------------------------------------------------------------------------
# Feature Matrix Builder
# -----------------------------------------------------------------------------

def build_feature_matrix(
    sample_sheet: pd.DataFrame,
    samples_dir: str = SAMPLES_DIR,
    save_interim: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Iterate over all samples in the sample sheet, load their TPM values,
    and merge into a single feature matrix.

    Args:
        sample_sheet : DataFrame from load_sample_sheet().
        samples_dir  : Root directory containing sample subfolders.
        save_interim : Whether to save the raw feature matrix to disk.

    Returns:
        Tuple of:
            - X : Feature matrix (samples × genes), rows=samples, cols=genes
            - y : Binary label Series (0=Normal, 1=Tumor)
    """
    logger.info("Building feature matrix from raw TSV files...")
    logger.info(f"Total samples to process: {len(sample_sheet)}")

    tpm_series_list = []
    valid_file_ids  = []
    gene_name_map   = {}

    for _, row in tqdm(sample_sheet.iterrows(), total=len(sample_sheet), desc="Loading samples"):
        file_id        = row["file_id"]
        result         = load_single_tsv(file_id, samples_dir)

        if result is not None:
            series, gene_map = result
            tpm_series_list.append(series)
            valid_file_ids.append(file_id)
            # Collect gene name mapping from first successful load
            if not gene_name_map and gene_map:
                gene_name_map = gene_map

    if len(tpm_series_list) == 0:
        raise RuntimeError("No samples were successfully loaded. Check your SAMPLES_DIR path.")

    logger.info(f"Successfully loaded: {len(tpm_series_list)} / {len(sample_sheet)} samples.")

    # Merge all series into a matrix (samples × genes)
    X = pd.DataFrame(tpm_series_list)
    X.index = valid_file_ids

    # Align labels to successfully loaded samples
    label_map_series = sample_sheet.set_index("file_id")["label"]
    y = label_map_series.loc[valid_file_ids]

    # Report any missing values
    missing = X.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Missing values detected: {missing} — filling with 0.")
        X = X.fillna(0)

    log_dataframe_info(X, "Feature matrix (raw)", logger)
    check_class_balance(y, logger)

    # Save gene name mapping
    if gene_name_map:
        import json
        ensure_dir(os.path.dirname(GENE_NAME_MAP_PATH))
        with open(GENE_NAME_MAP_PATH, "w") as f:
            json.dump(gene_name_map, f, indent=2)
        logger.info(f"Gene name mapping saved to: {GENE_NAME_MAP_PATH}")

    # Save interim feature matrix
    if save_interim:
        logger.info(f"Saving raw feature matrix to: {FEATURE_MATRIX_RAW}")
        save_dataframe(X, FEATURE_MATRIX_RAW, index=True)

    return X, y


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def load_data(
    samples_dir: str = SAMPLES_DIR,
    save_interim: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full data loading pipeline:
        1. Load and parse the GDC sample sheet
        2. Load TPM values from all RNA-seq TSV files
        3. Build and return the feature matrix and labels

    Args:
        samples_dir  : Root directory containing sample subfolders.
        save_interim : Whether to save the raw feature matrix to disk.

    Returns:
        Tuple of:
            - X : Feature matrix (samples × genes)
            - y : Binary label Series (0=Normal, 1=Tumor)
    """
    logger.info("=" * 60)
    logger.info("DATA LOADING STARTED")
    logger.info("=" * 60)

    sample_sheet = load_sample_sheet(SAMPLE_SHEET_PATH)
    X, y         = build_feature_matrix(sample_sheet, samples_dir, save_interim)

    logger.info("DATA LOADING COMPLETED")
    logger.info("=" * 60)

    return X, y
