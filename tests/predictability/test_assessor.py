import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.propensity.predictability.assessor import run_ten_fold_cv


def _fixture_Xy(n_samples=200, n_informative=4, seed=0):
    X_arr, y_arr = make_classification(
        n_samples=n_samples, n_features=6, n_informative=n_informative,
        n_redundant=0, random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr)
    return X, y


def test_run_ten_fold_cv_returns_auroc_in_valid_range():
    X, y = _fixture_Xy()
    auroc = run_ten_fold_cv(X, y)
    assert 0.0 <= auroc <= 1.0


def test_run_ten_fold_cv_is_deterministic_for_same_seed():
    X, y = _fixture_Xy()
    first = run_ten_fold_cv(X, y, seed=42)
    second = run_ten_fold_cv(X, y, seed=42)
    assert first == second


def test_run_ten_fold_cv_informative_features_beat_pure_noise():
    """A dataset with genuinely predictive features should score well above
    chance (0.5); this is a sanity check the pipeline isn't silently
    scrambling labels/predictions rather than a tight numeric assertion."""
    X, y = _fixture_Xy(n_samples=300, n_informative=5, seed=1)
    auroc = run_ten_fold_cv(X, y)
    assert auroc > 0.7


def test_run_ten_fold_cv_pure_noise_is_near_chance():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 2, size=300))
    auroc = run_ten_fold_cv(X, y)
    assert 0.3 < auroc < 0.7
