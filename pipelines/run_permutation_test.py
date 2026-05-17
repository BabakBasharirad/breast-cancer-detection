# =============================================================================
# pipelines/run_permutation_test.py
# -----------------------------------------------------------------------------
# Performs a permutation test to assess the statistical significance of the
# final model's performance.
#
# Procedure:
#   1. Load the final trained model and test set from disk
#   2. Record the real AUC-ROC score on the test set
#   3. Repeat N times:
#       a. Randomly shuffle the test labels
#       b. Compute AUC-ROC on shuffled labels
#   4. Compute p-value: fraction of permuted scores >= real score
#   5. Save results and plot permutation distribution
#
# Requires: run_training.py and run_evaluation.py to have been run first.
#
# Usage:
#   Run directly in VS Code or terminal:
#   python pipelines/run_permutation_test.py
# =============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from utils   import setup_logger, ensure_dir, load_dataframe
from model   import load_model

logger = setup_logger(
    "run_permutation_test",
    log_file=os.path.join(config.RESULTS_DIR, "permutation_test.log")
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
TEST_X_PATH           = os.path.join(config.PROCESSED_DIR, "X_test.csv")
TEST_Y_PATH           = os.path.join(config.PROCESSED_DIR, "y_test.csv")
PERMUTATION_JSON_PATH = os.path.join(config.METRICS_DIR,  "permutation_test.json")
PERMUTATION_PLOT_PATH = os.path.join(config.FIGURES_DIR,  "permutation_test.png")

# Number of permutations
N_PERMUTATIONS = 1000


def run_permutation_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """
    Run a permutation test on the test set.

    Args:
        model          : Trained final RandomForestClassifier.
        X_test         : Test feature matrix.
        y_test         : True test labels.
        n_permutations : Number of permutations to perform.

    Returns:
        Dictionary containing real score, permuted scores, and p-value.
    """
    logger.info("=" * 60)
    logger.info("PERMUTATION TEST STARTED")
    logger.info(f"Number of permutations: {n_permutations}")
    logger.info("=" * 60)

    # Real score
    y_prob     = model.predict_proba(X_test)[:, 1]
    real_score = roc_auc_score(y_test, y_prob)
    logger.info(f"Real AUC-ROC: {real_score:.4f}")

    # Permuted scores
    permuted_scores = []
    rng = np.random.RandomState(config.RANDOM_STATE)

    for i in range(n_permutations):
        y_shuffled     = rng.permutation(y_test.values)
        permuted_score = roc_auc_score(y_shuffled, y_prob)
        permuted_scores.append(permuted_score)

        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i + 1}/{n_permutations} permutations...")

    permuted_scores = np.array(permuted_scores)

    # p-value: fraction of permuted scores >= real score
    p_value = np.mean(permuted_scores >= real_score)

    logger.info("=" * 60)
    logger.info("PERMUTATION TEST RESULTS")
    logger.info(f"  Real AUC-ROC          : {real_score:.4f}")
    logger.info(f"  Mean permuted AUC-ROC : {permuted_scores.mean():.4f}")
    logger.info(f"  Std permuted AUC-ROC  : {permuted_scores.std():.4f}")
    logger.info(f"  p-value               : {p_value:.4f}")
    logger.info(f"  Significant (p<0.05)  : {p_value < 0.05}")
    logger.info("=" * 60)

    return {
        "real_score"          : round(float(real_score),            4),
        "mean_permuted_score" : round(float(permuted_scores.mean()), 4),
        "std_permuted_score"  : round(float(permuted_scores.std()),  4),
        "p_value"             : round(float(p_value),               4),
        "n_permutations"      : n_permutations,
        "significant"         : bool(p_value < 0.05),
    }


def plot_permutation_distribution(
    real_score: float,
    permuted_scores: np.ndarray,
    p_value: float,
    save_path: str = PERMUTATION_PLOT_PATH,
) -> None:
    """
    Plot the distribution of permuted AUC-ROC scores against the real score.

    Args:
        real_score      : Real model AUC-ROC score.
        permuted_scores : Array of permuted AUC-ROC scores.
        p_value         : Computed p-value.
        save_path       : File path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot permuted score distribution
    ax.hist(
        permuted_scores,
        bins=40,
        color="steelblue",
        alpha=0.75,
        edgecolor="white",
        label=f"Permuted scores (n={len(permuted_scores)})",
    )

    # Plot real score as vertical line
    ax.axvline(
        real_score,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"Real AUC-ROC = {real_score:.4f}",
    )

    ax.set_xlabel("AUC-ROC Score",       fontsize=12)
    ax.set_ylabel("Frequency",           fontsize=12)
    ax.set_title(
        f"Permutation Test — AUC-ROC Distribution\n"
        f"p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not Significant'})",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="png", bbox_inches="tight")
    plt.close()

    logger.info(f"Permutation test plot saved to: {save_path}")


def main():
    logger.info("Loading test set and model...")
    X_test = load_dataframe(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH, index_col=0).squeeze()
    model  = load_model()

    # Run permutation test
    results = run_permutation_test(model, X_test, y_test, N_PERMUTATIONS)

    # Recompute permuted scores for plotting
    y_prob  = model.predict_proba(X_test)[:, 1]
    rng     = np.random.RandomState(config.RANDOM_STATE)
    permuted_scores = np.array([
        roc_auc_score(rng.permutation(y_test.values), y_prob)
        for _ in range(N_PERMUTATIONS)
    ])

    # Plot
    plot_permutation_distribution(
        real_score      = results["real_score"],
        permuted_scores = permuted_scores,
        p_value         = results["p_value"],
    )

    # Save results
    ensure_dir(config.METRICS_DIR)
    with open(PERMUTATION_JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Permutation test results saved to: {PERMUTATION_JSON_PATH}")

    logger.info("Permutation test complete.")


if __name__ == "__main__":
    main()
