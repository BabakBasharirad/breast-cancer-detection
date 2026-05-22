# =============================================================================
# balancer.py
# -----------------------------------------------------------------------------
# Handles class imbalance between tumor (majority) and normal (minority)
# samples. Implements and compares multiple balancing strategies.
#
# Strategies:
#   1. none          — No balancing (baseline)
#   2. class_weight  — Built-in RF class weighting (no resampling)
#   3. random_over   — Random Oversampling of minority class
#   4. random_under  — Random Undersampling of majority class
#   5. smote         — Synthetic Minority Over-sampling Technique
#   6. smoteenn      — SMOTE + Edited Nearest Neighbours (cleaning)
#   7. smotetomek    — SMOTE + Tomek Links (cleaning)
#
# IMPORTANT: Balancing is applied ONLY on training data — never on test data.
# =============================================================================

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from config import RANDOM_STATE
from utils import setup_logger, check_class_balance

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Available Strategies
# -----------------------------------------------------------------------------

# Strategies that require resampling (return new X, y)
RESAMPLING_STRATEGIES = [
    "none",
    "random_over",
    "random_under",
    "smote",
    "smoteenn",
    "smotetomek",
]

# Strategies handled internally by the model (no resampling needed)
MODEL_WEIGHT_STRATEGIES = [
    "class_weight",
]

ALL_STRATEGIES = RESAMPLING_STRATEGIES + MODEL_WEIGHT_STRATEGIES


# -----------------------------------------------------------------------------
# Individual Strategy Functions
# -----------------------------------------------------------------------------

def apply_random_oversampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly duplicate minority class (normal) samples until classes
    are balanced.

    Args:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def apply_random_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly remove majority class (tumor) samples until classes
    are balanced.

    Args:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).
    Generates synthetic normal samples by interpolating between
    existing minority class samples in feature space.

    Args:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def apply_smoteenn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE + Edited Nearest Neighbours (ENN).
    Combines oversampling of minority class with cleaning of
    ambiguous samples near the decision boundary.

    Args:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    X_res, y_res = smoteenn.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def apply_smotetomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE + Tomek Links.
    Combines oversampling of minority class with removal of
    Tomek links (borderline majority samples) for cleaner boundaries.

    Args:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    smotetomek = SMOTETomek(random_state=RANDOM_STATE)
    X_res, y_res = smotetomek.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


# -----------------------------------------------------------------------------
# Main Balancing Function
# -----------------------------------------------------------------------------

def apply_balancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply the specified balancing strategy to the training data.

    For 'none' and 'class_weight', the original data is returned unchanged.
    The 'class_weight' strategy is handled directly inside the Random Forest
    model via RF_PARAMS in config.py.

    Args:
        X_train  : Training feature matrix.
        y_train  : Training labels.
        strategy : Balancing strategy name (see ALL_STRATEGIES).

    Returns:
        Tuple of resampled (X_train, y_train).
        For 'none' and 'class_weight', original data is returned unchanged.
    """
    logger.info(f"Applying balancing strategy: '{strategy}'")

    if strategy not in ALL_STRATEGIES:
        raise ValueError(
            f"Unknown balancing strategy: '{strategy}'. "
            f"Choose from: {ALL_STRATEGIES}"
        )

    # Log class distribution before balancing
    logger.info("Class distribution BEFORE balancing:")
    check_class_balance(y_train, logger)

    if strategy == "none":
        logger.info("  No balancing applied.")
        return X_train, y_train

    elif strategy == "class_weight":
        logger.info("  class_weight strategy — handled by Random Forest internally.")
        logger.info("  No resampling applied to training data.")
        return X_train, y_train

    elif strategy == "random_over":
        X_res, y_res = apply_random_oversampling(X_train, y_train)

    elif strategy == "random_under":
        X_res, y_res = apply_random_undersampling(X_train, y_train)

    elif strategy == "smote":
        X_res, y_res = apply_smote(X_train, y_train)

    elif strategy == "smoteenn":
        X_res, y_res = apply_smoteenn(X_train, y_train)

    elif strategy == "smotetomek":
        X_res, y_res = apply_smotetomek(X_train, y_train)

    # Log class distribution after balancing
    logger.info("Class distribution AFTER balancing:")
    check_class_balance(y_res, logger)

    # Log post-resampling counts
    n_tumor  = (y_res == 1).sum()
    n_normal = (y_res == 0).sum()
    logger.info(f"  Post-resampling: Total={len(y_res)}, Tumor={n_tumor}, Normal={n_normal}")

    return X_res, y_res


# -----------------------------------------------------------------------------
# Get RF class_weight Parameter
# -----------------------------------------------------------------------------

def get_class_weight_param(strategy: str) -> str | None:
    """
    Return the appropriate class_weight parameter for the Random Forest
    based on the balancing strategy.

    For 'class_weight' strategy, returns 'balanced' to let RF handle
    imbalance internally. For all other strategies, returns None
    (resampling already handled externally).

    Args:
        strategy : Balancing strategy name.

    Returns:
        'balanced' for class_weight strategy, None otherwise.
    """
    return "balanced" if strategy == "class_weight" else None
