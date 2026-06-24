"""
Machine Learning Pipeline for Knee Osteoarthritis Prediction

This script implements a two-stage classification framework:
1) Binary classification: KL = 0 vs KL ∈ {2, 4}
2) Secondary classification: KL = 2 vs KL = 4

Dataset:
- Osteoarthritis Initiative (OAI)-based features
- Finite element-derived and radiographic predictors

Outputs:
- Weighted F1-score
- Balanced accuracy
- Confusion matrix (aggregated across folds)

Author: Teemu Nurmirinta
Journal: CMBBE
Year: 2026
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier


# =========================
# Configuration
# =========================
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 25
N_TREES = 3000

FEATURES = [
    'Age', 'Weight', 'Height',
    'Lateral joint space', 'Medial joint space',
    'Femur width', 'Intercondylar distance',
    'Femur-tibia angle', 'Varus-valgus angle',
    'Proximal tibia width', 'Distal tibia width',
    'KL Baseline'
]

TARGET = 'KL Max'


# =========================
# Data Loading
# =========================
def load_data(filepath):
    """
    Load dataset and prepare features and targets.

    Parameters
    ----------
    filepath : str
        Path to CSV dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    y_binary : np.ndarray
        Binary labels for stage 1
    """
    dataset = pd.read_csv(filepath)

    X = dataset[FEATURES].values
    y = dataset[TARGET].values

    # Stage 1 labels: 0 vs {2,4}
    y_binary = np.where(np.isin(y, [2, 4]), 1, 0)

    return X, y, y_binary


# =========================
# Two-Stage Model
# =========================
def two_stage_model(X, y, y_binary, train_idx, test_idx):
    """
    Two-stage classification:
    Stage 1: KL 0 vs KL {2,4}
    Stage 2: KL 2 vs KL 4

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """

    # ---- Stage 1 ----
    clf_stage1 = BalancedRandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf_stage1.fit(X[train_idx], y_binary[train_idx])
    y_pred_stage1 = clf_stage1.predict(X[test_idx])

    # ---- Stage 2 ----
    X_train_min = X[train_idx][y_binary[train_idx] == 1]
    y_train_min = y[train_idx][y_binary[train_idx] == 1]

    clf_stage2 = BalancedRandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf_stage2.fit(X_train_min, y_train_min)

    X_test_min = X[test_idx][y_pred_stage1 == 1]

    # Handle edge case: no minority samples
    if len(X_test_min) == 0:
        return y[test_idx], y_pred_stage1

    y_pred_stage2 = clf_stage2.predict(X_test_min)

    # ---- Combine predictions ----
    y_pred_final = y_pred_stage1.copy()
    y_pred_final[y_pred_stage1 == 1] = y_pred_stage2

    return y[test_idx], y_pred_final


# =========================
# Cross-Validation
# =========================
def run_experiment(X, y, y_binary):
    """
    Perform repeated stratified cross-validation.

    Returns
    -------
    dict with metrics
    """

    f1_scores = []
    ba_scores = []
    conf_matrix = np.zeros((4, 4))

    for repeat in range(N_REPEATS):

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=repeat  # ensures reproducibility
        )

        for train_idx, test_idx in skf.split(X, y):

            y_true, y_pred = two_stage_model(
                X, y, y_binary, train_idx, test_idx
            )

            f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
            ba_scores.append(balanced_accuracy_score(y_true, y_pred))

            conf_matrix += confusion_matrix(
                y_true, y_pred,
                labels=[0, 1, 2, 4]
            )

    return {
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "ba_mean": np.mean(ba_scores),
        "ba_std": np.std(ba_scores),
        "conf_matrix_total": conf_matrix,
        "conf_matrix_avg": conf_matrix / N_REPEATS
    }


# =========================
# Main Execution
# =========================
def main():
    """
    Main function to run the experiment.
    """

    FILEPATH = "data.csv"  # <-- replace with your dataset

    print("Loading data...")
    X, y, y_binary = load_data(FILEPATH)

    print("Running experiment...")
    results = run_experiment(X, y, y_binary)

    print("\n=== Results ===")
    print(f"Weighted F1-score: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    print(f"Balanced Accuracy: {results['ba_mean']:.3f} ± {results['ba_std']:.3f}")

    print("\nConfusion Matrix (Total):")
    print(results["conf_matrix_total"])

    print("\nConfusion Matrix (Average per repeat):")
    print(results["conf_matrix_avg"])


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main()