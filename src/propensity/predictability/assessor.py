"""Random Forest assessor evaluation: predicts a binary outcome (e.g. model
correctness) from an arbitrary feature matrix via 10-fold cross-validation.

RandomForestClassifier hyperparameters (min_samples_split=50 + entropy
criterion) match prop-meas-anon/scripts/inference/assessors.py's
run_ten_fold_ID_evaluation. AUROC aggregation deliberately does NOT match
that original code, though: the original pools every fold's held-out
predictions into one array and computes a single roc_auc_score over the
pool. That treats predictions from 10 different fitted models as directly
comparable rankings, which they aren't -- each fold fits its own Random
Forest on different training data, so the resulting score isn't the
generalisation performance of one model. The methodologically sound
aggregation is the mean of each fold's own AUROC (computed against only
that fold's model and held-out data), which is what this implementation
does instead.

This is the one evaluation method implemented for now; a bootstrap-CI
evaluator is a planned sibling addition without changing this signature.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def run_ten_fold_cv(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    """Fits a fresh RandomForestClassifier per fold over a 10-fold KFold
    split and returns the mean of each fold's own AUROC (computed from that
    fold's held-out predictions against that fold's model only -- never
    pooled across folds, since different folds fit different models).

    A fold whose held-out set contains only one class is skipped (AUROC is
    undefined there) rather than counted as an error; if every fold is
    skipped this way (e.g. y has only one class overall), returns NaN.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    fold_aurocs = []

    for train_idx, test_idx in kf.split(X):
        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue
        model = RandomForestClassifier(random_state=seed, min_samples_split=50, criterion="entropy")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        fold_aurocs.append(roc_auc_score(y_test, y_pred_proba))

    return float(np.mean(fold_aurocs)) if fold_aurocs else float("nan")
