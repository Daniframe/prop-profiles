import numpy as np
import pytest

from src.propensity.surfaces.model import two_sided_sigma
from src.propensity.surfaces.mle import fit_theta, neg_log_likelihood


def _simulate(theta_true, n_items=250, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 2.5, n_items)
    x1 = centers - half_widths
    x2 = centers + half_widths
    demands = np.column_stack([x1, x2])

    probs = np.array([two_sided_sigma(theta_true, a, b, 1.0, 1.0) for a, b in demands])
    success = rng.binomial(1, probs)
    return demands, success


def test_fit_theta_recovers_known_theta():
    theta_true = -1.5
    demands, success = _simulate(theta_true, seed=1)
    fit = fit_theta(demands, success)
    assert abs(fit["theta_hat"] - theta_true) < 0.4


def test_ci_bounds_ordering():
    demands, success = _simulate(0.5, seed=2)
    fit = fit_theta(demands, success)
    assert fit["ci95_lower"] < fit["theta_hat"] < fit["ci95_upper"]


def test_neg_log_likelihood_matches_hand_computation():
    demands = np.array([[-1.0, 1.0], [0.0, 2.0]])
    success = np.array([1, 0])

    p0 = two_sided_sigma(0.3, -1.0, 1.0, 1.0, 1.0)
    p1 = two_sided_sigma(0.3, 0.0, 2.0, 1.0, 1.0)
    expected = -np.log(p0 * (1 - p1))

    assert neg_log_likelihood(0.3, demands, success) == pytest.approx(expected, rel=1e-9)
