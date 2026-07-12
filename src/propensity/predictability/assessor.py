"""Random Forest assessor evaluation: predicts a binary outcome (e.g. model
correctness) from an arbitrary feature matrix via 10-fold cross-validation.

Ported from prop-meas-anon/scripts/inference/assessors.py's
run_ten_fold_ID_evaluation, matching its exact methodology (plain
non-stratified KFold, min_samples_split=50 + entropy criterion, and AUROC
computed once over pooled out-of-fold predictions rather than averaged
per-fold) so results are comparable to the paper's Table 2.

This is the one evaluation method implemented for now; a different CV
scheme or a bootstrap-CI evaluator could be added later as a sibling
function without changing this one's signature.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def run_ten_fold_cv(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    """Fits a fresh RandomForestClassifier per fold over a 10-fold KFold
    split, pools every fold's held-out predicted probabilities, and returns
    a single AUROC over that pooled out-of-fold set (not a mean of 10
    per-fold AUROCs -- this matches the original pipeline's aggregation).
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    pooled_true = np.empty(len(y))
    pooled_pred = np.empty(len(y))

    for train_idx, test_idx in kf.split(X):
        model = RandomForestClassifier(random_state=seed, min_samples_split=50, criterion="entropy")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pooled_pred[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        pooled_true[test_idx] = y.iloc[test_idx].values

    return roc_auc_score(pooled_true, pooled_pred)
