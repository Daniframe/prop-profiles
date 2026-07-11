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


def _simulate_near_ceiling(n_items=300, seed=0, n_failures=15):
    """Near-degenerate data: almost every item succeeds regardless of demand
    interval, with only a handful of scattered failures -- the same failure
    mode observed in the real Ul benchmark's unstable (model, level) fits,
    where the likelihood surface is nearly flat and a single BFGS run can
    land in an arbitrary local optimum."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 1.5, n_items)
    demands = np.column_stack([centers - half_widths, centers + half_widths])
    success = np.ones(n_items, dtype=int)
    fail_idx = rng.choice(n_items, size=n_failures, replace=False)
    success[fail_idx] = 0
    return demands, success


def test_fit_theta_default_n_restarts_returns_same_keys_as_before():
    demands, success = _simulate(0.0, seed=3)
    fit = fit_theta(demands, success)
    assert set(fit.keys()) == {
        "theta_hat", "se", "ci95_lower", "ci95_upper",
        "convergence", "reference_ll", "gof", "pseudo_r2",
    }


def test_fit_theta_n_restarts_greater_than_one_adds_diagnostic_keys():
    demands, success = _simulate(0.0, seed=3)
    fit = fit_theta(demands, success, n_restarts=5)
    assert {"n_restarts", "n_converged", "restart_theta_std"} <= set(fit.keys())
    assert fit["n_restarts"] == 5


def test_fit_theta_multistart_is_never_worse_than_single_start():
    """The single-start x_init is always included among the multi-start
    candidates, so the best-likelihood-among-converged selection can only
    match or beat (lower gof/nll, i.e. higher or equal likelihood) the
    single-start result -- never worse."""
    demands, success = _simulate_near_ceiling(seed=7)
    single = fit_theta(demands, success)
    multi = fit_theta(demands, success, n_restarts=6)
    assert multi["gof"] >= single["gof"] - 1e-9


def test_fit_theta_multistart_reports_convergence_spread_diagnostic():
    demands, success = _simulate_near_ceiling(seed=11)
    multi = fit_theta(demands, success, n_restarts=6)
    assert multi["n_converged"] <= multi["n_restarts"]
    assert multi["restart_theta_std"] >= 0.0
