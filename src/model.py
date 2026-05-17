# =============================================================================
# model.py
# -----------------------------------------------------------------------------
# Handles Random Forest model training, cross-validation across all balancing
# strategies, best strategy selection, and final model training.
#
# Pipeline:
#   1. For each balancing strategy:
#       a. Apply balancing to training data
#       b. Run stratified k-fold cross-validation
#       c. Record CV metrics
#   2. Compare strategies and select the best one
#   3. Retrain final model on full training set with best strategy
#   4. Save the final model
# =============================================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from config import (
    RF_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
    METRICS_DIR,
    RESULTS_DIR,
)
from balancer import apply_balancing, get_class_weight_param, ALL_STRATEGIES
from utils import setup_logger, ensure_dir, check_class_balance

logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
CV_RESULTS_PATH      = os.path.join(METRICS_DIR, "cv_results_all_strategies.csv")
BEST_STRATEGY_PATH   = os.path.join(METRICS_DIR, "best_strategy.json")
FINAL_MODEL_PATH     = os.path.join(RESULTS_DIR, "final_model.pkl")

# -----------------------------------------------------------------------------
# Scoring Metrics for Cross-Validation
# -----------------------------------------------------------------------------
CV_SCORING = {
    "accuracy" : make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall"   : make_scorer(recall_score,    zero_division=0),
    "f1"       : make_scorer(f1_score,        zero_division=0),
    "roc_auc"  : "roc_auc",
}

# Metric used to select the best balancing strategy
BEST_STRATEGY_METRIC = "roc_auc"


# -----------------------------------------------------------------------------
# Build Random Forest
# -----------------------------------------------------------------------------

def build_random_forest(class_weight: str | None = None) -> RandomForestClassifier:
    """
    Instantiate a Random Forest classifier using parameters from config.py.
    The class_weight parameter is overridden based on the balancing strategy.

    Args:
        class_weight : 'balanced' for class_weight strategy, None otherwise.

    Returns:
        Configured RandomForestClassifier instance.
    """
    params = RF_PARAMS.copy()
    params["class_weight"] = class_weight
    return RandomForestClassifier(**params)


# -----------------------------------------------------------------------------
# Cross-Validation for a Single Strategy
# -----------------------------------------------------------------------------

def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
) -> dict:
    """
    Apply a balancing strategy and run stratified k-fold cross-validation
    on the training set.

    Args:
        X_train  : Training feature matrix.
        y_train  : Training labels.
        strategy : Balancing strategy name.

    Returns:
        Dictionary of mean and std CV metrics for this strategy.
    """
    logger.info(f"Running {CV_FOLDS}-fold CV for strategy: '{strategy}'...")

    # Apply balancing (on training data only — CV handles splits internally)
    X_bal, y_bal = apply_balancing(X_train, y_train, strategy)

    # Get appropriate class_weight for RF
    class_weight = get_class_weight_param(strategy)
    model        = build_random_forest(class_weight=class_weight)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = cross_validate(
        model,
        X_bal,
        y_bal,
        cv=skf,
        scoring=CV_SCORING,
        return_train_score=False,
        n_jobs=-1,
    )

    # Summarize results
    summary = {"strategy": strategy}
    for metric in CV_SCORING.keys():
        scores = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = round(float(np.mean(scores)), 4)
        summary[f"{metric}_std"]  = round(float(np.std(scores)),  4)

    logger.info(
        f"  AUC: {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f} | "
        f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f} | "
        f"Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}"
    )

    return summary


# -----------------------------------------------------------------------------
# Compare All Balancing Strategies
# -----------------------------------------------------------------------------

def compare_balancing_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategies: list = ALL_STRATEGIES,
) -> pd.DataFrame:
    """
    Run cross-validation for all balancing strategies and compile
    a comparison table.

    Args:
        X_train    : Training feature matrix.
        y_train    : Training labels.
        strategies : List of balancing strategy names to compare.

    Returns:
        DataFrame with CV metrics for all strategies, sorted by AUC.
    """
    logger.info("=" * 60)
    logger.info("COMPARING BALANCING STRATEGIES")
    logger.info(f"Strategies: {strategies}")
    logger.info("=" * 60)

    results = []
    for strategy in strategies:
        summary = run_cross_validation(X_train, y_train, strategy)
        results.append(summary)

    # Build comparison DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by=f"{BEST_STRATEGY_METRIC}_mean",
        ascending=False,
    ).reset_index(drop=True)

    # Save comparison table
    ensure_dir(METRICS_DIR)
    df_results.to_csv(CV_RESULTS_PATH, index=False)
    logger.info(f"CV results saved to: {CV_RESULTS_PATH}")

    # Log full comparison table
    logger.info("\nBalancing Strategy Comparison (sorted by AUC):")
    logger.info("\n" + df_results.to_string(index=False))

    return df_results


# -----------------------------------------------------------------------------
# Select Best Strategy
# -----------------------------------------------------------------------------

def select_best_strategy(cv_results: pd.DataFrame) -> str:
    """
    Select the best balancing strategy using a multi-criteria tie-breaking
    approach based on cross-validation results only (no test set used).

    Tie-breaking order:
        1. Highest CV AUC-ROC mean (primary metric)
        2. Highest CV F1 mean
        3. Lowest CV AUC-ROC std (most stable)
        4. Highest CV Recall mean (important in medical classification)

    Args:
        cv_results : DataFrame from compare_balancing_strategies().

    Returns:
        Name of the best balancing strategy.
    """
    df = cv_results.copy()

    # Sort by multiple criteria: AUC desc, F1 desc, AUC std asc, Recall desc
    df = df.sort_values(
        by=["roc_auc_mean", "f1_mean", "roc_auc_std", "recall_mean"],
        ascending=[False, False, True, False]
    ).reset_index(drop=True)

    best_row      = df.iloc[0]
    best_strategy = best_row["strategy"]
    best_auc      = best_row["roc_auc_mean"]
    best_f1       = best_row["f1_mean"]
    best_auc_std  = best_row["roc_auc_std"]
    best_recall   = best_row["recall_mean"]

    logger.info("=" * 60)
    logger.info(f"BEST STRATEGY: '{best_strategy}'")
    logger.info(f"  AUC-ROC : {best_auc:.4f} (std: {best_auc_std:.4f})")
    logger.info(f"  F1 Score: {best_f1:.4f}")
    logger.info(f"  Recall  : {best_recall:.4f}")
    logger.info("  Selection criteria: AUC-ROC > F1 > AUC std > Recall")
    logger.info("=" * 60)

    # Save best strategy info
    best_info = {
        "best_strategy"   : best_strategy,
        "selection_metric": "multi-criteria (AUC-ROC, F1, AUC std, Recall)",
        "roc_auc_mean"    : best_auc,
        "roc_auc_std"     : best_auc_std,
        "f1_mean"         : best_f1,
        "recall_mean"     : best_recall,
        "all_results"     : df.to_dict(orient="records"),
    }

    ensure_dir(METRICS_DIR)
    with open(BEST_STRATEGY_PATH, "w") as f:
        json.dump(best_info, f, indent=4)
    logger.info(f"Best strategy info saved to: {BEST_STRATEGY_PATH}")

    return best_strategy


# -----------------------------------------------------------------------------
# Train Final Model
# -----------------------------------------------------------------------------

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_strategy: str,
) -> RandomForestClassifier:
    """
    Retrain the final Random Forest model on the full training set
    using the best balancing strategy identified from cross-validation.

    Args:
        X_train       : Full training feature matrix.
        y_train       : Full training labels.
        best_strategy : Best balancing strategy name.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING FINAL MODEL with strategy: '{best_strategy}'")
    logger.info("=" * 60)

    # Apply best balancing strategy
    X_bal, y_bal = apply_balancing(X_train, y_train, best_strategy)

    logger.info("Final training set class distribution:")
    check_class_balance(y_bal, logger)

    # Build and train model
    class_weight = get_class_weight_param(best_strategy)
    model        = build_random_forest(class_weight=class_weight)

    logger.info(f"Training Random Forest with {RF_PARAMS['n_estimators']} trees...")
    model.fit(X_bal, y_bal)
    logger.info("Final model training completed.")

    # Save model to disk
    _save_model(model)

    return model


# -----------------------------------------------------------------------------
# Save / Load Model
# -----------------------------------------------------------------------------

def _save_model(model: RandomForestClassifier) -> None:
    """
    Serialize and save the trained model to disk using pickle.

    Args:
        model : Trained RandomForestClassifier.
    """
    ensure_dir(os.path.dirname(FINAL_MODEL_PATH))
    with open(FINAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Final model saved to: {FINAL_MODEL_PATH}")


def load_model(path: str = FINAL_MODEL_PATH) -> RandomForestClassifier:
    """
    Load a previously saved model from disk.

    Args:
        path : Path to the saved model pickle file.

    Returns:
        Loaded RandomForestClassifier.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from: {path}")
    return model


# -----------------------------------------------------------------------------
# Full Model Pipeline
# -----------------------------------------------------------------------------

def run_model_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategies: list = ALL_STRATEGIES,
) -> tuple[RandomForestClassifier, str, pd.DataFrame]:
    """
    Full model pipeline:
        1. Compare all balancing strategies via cross-validation
        2. Select the best strategy
        3. Train the final model on full training set

    Args:
        X_train    : Training feature matrix.
        y_train    : Training labels.
        strategies : List of balancing strategies to compare.

    Returns:
        Tuple of:
            - final_model   : Trained RandomForestClassifier
            - best_strategy : Name of the best balancing strategy
            - cv_results    : Full CV comparison DataFrame
    """
    # Step 1: Compare all strategies
    cv_results    = compare_balancing_strategies(X_train, y_train, strategies)

    # Step 2: Select best strategy
    best_strategy = select_best_strategy(cv_results)

    # Step 3: Train final model
    final_model   = train_final_model(X_train, y_train, best_strategy)

    return final_model, best_strategy, cv_results
