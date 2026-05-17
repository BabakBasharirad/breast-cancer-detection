# =============================================================================
# utils.py
# -----------------------------------------------------------------------------
# Shared utility functions used across the pipeline.
# Includes logging setup, directory creation, and helper functions.
# =============================================================================

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name      : Logger name (typically the module name).
        log_file  : Optional path to a log file.
        level     : Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Directory Management
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """
    Create a directory (and any parent directories) if it does not exist.

    Args:
        path: Directory path to create.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_project_dirs(config) -> None:
    """
    Ensure all required project directories exist.
    Accepts the config module and creates all output directories.

    Args:
        config: The config module (src.config).
    """
    dirs = [
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.FIGURES_DIR,
        config.METRICS_DIR,
    ]
    for d in dirs:
        ensure_dir(d)


# -----------------------------------------------------------------------------
# Data I/O Helpers
# -----------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: str, index: bool = True) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df    : DataFrame to save.
        path  : Destination file path.
        index : Whether to write the index (default: True).
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)


def load_dataframe(path: str, index_col: int = 0) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Args:
        path      : Source file path.
        index_col : Column to use as index (default: 0).

    Returns:
        Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, index_col=index_col)


def save_metrics(metrics: dict, path: str) -> None:
    """
    Save evaluation metrics dictionary to a JSON file.

    Args:
        metrics : Dictionary of metric names and values.
        path    : Destination file path.
    """
    ensure_dir(os.path.dirname(path))

    # Add timestamp
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics(path: str) -> dict:
    """
    Load evaluation metrics from a JSON file.

    Args:
        path: Source file path.

    Returns:
        Dictionary of metrics.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# General Helpers
# -----------------------------------------------------------------------------

def get_timestamp() -> str:
    """
    Return the current timestamp as a formatted string.

    Returns:
        Timestamp string in YYYY-MM-DD_HH-MM-SS format.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log_dataframe_info(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    """
    Log basic information about a DataFrame.

    Args:
        df     : DataFrame to describe.
        name   : Label to identify the DataFrame in the log.
        logger : Logger instance.
    """
    logger.info(f"{name} — shape: {df.shape} | memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")


def check_class_balance(labels: pd.Series, logger: logging.Logger) -> None:
    """
    Log the class distribution of the label series.

    Args:
        labels : Series of binary labels (0/1).
        logger : Logger instance.
    """
    counts = labels.value_counts()
    total  = len(labels)
    logger.info("Class distribution:")
    for label, count in counts.items():
        label_name = "Tumor (1)" if label == 1 else "Normal (0)"
        logger.info(f"  {label_name}: {count} ({count / total * 100:.1f}%)")


def load_gene_name_map(path: str) -> dict:
    """
    Load gene_id to gene_name mapping from JSON file.

    Args:
        path: Path to gene_name_mapping.json

    Returns:
        Dictionary mapping gene_id to gene_name.
    """
    import json
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gene name mapping not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def apply_gene_names(ids: list, gene_map: dict) -> list:
    """
    Replace Ensembl gene IDs with gene names where available.
    Falls back to original ID if name not found.

    Args:
        ids      : List of Ensembl gene IDs.
        gene_map : Dictionary mapping gene_id to gene_name.

    Returns:
        List of gene names.
    """
    return [gene_map.get(gid, gid) for gid in ids]


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for reproducibility.

    Args:
        seed: Integer seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
