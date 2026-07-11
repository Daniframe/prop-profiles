"""Eq. 6 maximum-likelihood fitting of a single propensity theta from
item-level (demand interval, outcome) data. Ported from get_prop_point.py.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess

from .model import two_sided_sigma


def neg_log_likelihood(theta, demands, success, k=1.0):
    """Eq. 6 negative log-likelihood: -sum_i [y_i log p_i + (1-y_i) log(1-p_i)],
    computed via the product form (probs**success * (1-probs)**(1-success))
    as in the original implementation, with probs clipped away from 0/1.
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)
    # scipy.optimize passes theta as a 1-element array; reduce to a plain
    # numpy scalar so two_sided_sigma's result can be assigned into probs[i]
    # (newer numpy no longer implicitly squeezes 1-element arrays there).
    theta = np.ravel(theta)[0]
    probs = np.zeros(len(demands))
    for i, (x1, x2) in enumerate(demands):
        probs[i] = two_sided_sigma(theta, x1, x2, k, k)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return -np.log(np.prod(probs**success * (1 - probs) ** (1 - success)))


def fit_theta(demands, success, k=1.0, x_init=None, n_bins=20, lowess_frac=0.4, maxiter=500,
              n_restarts=1, restart_range=None):
    """
    Fits theta by maximum likelihood (Eq. 6) via scipy.optimize.minimize
    (BFGS). If x_init is not given, derives one from a LOWESS-smoothed,
    binned empirical success curve (bin by interval center, smooth, take the
    argmax) -- exactly the approach used in get_prop_point.py.

    demands: (N, 2) array-like of (b_l, b_u) pairs.
    success: (N,) array-like of binary outcomes.
    n_restarts: if 1 (default), behaves exactly as a single BFGS run from the
        LOWESS-derived x_init (unchanged from before this parameter existed).
        If >1, also retries from (n_restarts - 1) additional starting points
        spread across restart_range, and keeps whichever attempt has the best
        (lowest) negative log-likelihood among ALL attempts, regardless of
        each attempt's own convergence flag (a non-converged run can still
        have reached a better point than a converged one elsewhere -- see the
        comment above the selection line). This matters when the likelihood
        surface is nearly flat (e.g. near-ceiling/near-floor outcome data): a
        single BFGS run can land in an arbitrary local optimum, and simply
        averaging thetas from different optima would blend two unrelated
        modes into a point that fits neither -- so this keeps the
        best-supported mode instead, not an average. Since the single-start
        attempt is always included among the candidates, more restarts can
        only match or improve the fit, never make it worse.
    restart_range: (low, high) to spread extra starting points across;
        defaults to the observed (min, max) of the demand bounds.

    Returns a dict: theta_hat, se, ci95_lower, ci95_upper, convergence,
    reference_ll, gof, pseudo_r2. When n_restarts > 1, also includes
    n_restarts, n_converged, and restart_theta_std (the spread of theta_hat
    across attempts -- a diagnostic for how ambiguous the fit is, not used
    to compute theta_hat itself).
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)

    if x_init is None:
        centers = (demands[:, 0] + demands[:, 1]) / 2
        bin_means, bin_edges, _ = binned_statistic(centers, success, statistic="mean", bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        finite = (~np.isnan(bin_means)) & (~np.isnan(bin_centers))
        lowess_smoothed = lowess(bin_means[finite], bin_centers[finite], frac=lowess_frac, it=0)
        idx_max = np.argmax(lowess_smoothed[:, 1])
        x_init = lowess_smoothed[idx_max, 0]

    if n_restarts > 1:
        if restart_range is None:
            restart_range = (float(demands.min()), float(demands.max()))
        starting_points = [x_init] + list(np.linspace(restart_range[0], restart_range[1], n_restarts - 1))
    else:
        starting_points = [x_init]

    attempts = [
        minimize(neg_log_likelihood, x0=sp, args=(demands, success, k), method="BFGS", options={"maxiter": maxiter})
        for sp in starting_points
    ]

    converged = [r for r in attempts if r.success]
    # Deliberately NOT pre-filtered to converged-only: a non-converged BFGS run
    # can still have reached a lower (better) NLL than every converged
    # alternative (e.g. it was still improving when maxiter cut it off), and
    # discarding it in favor of a worse-but-converged run would both give a
    # worse point estimate AND break the invariant that more restarts can
    # only match or improve on a single-start fit (the single-start attempt,
    # x_init, is always included in `attempts`). `convergence` in the
    # returned dict reflects whether the winning attempt itself converged,
    # so an unreliable winner is still flagged rather than silently hidden.
    best = min(attempts, key=lambda r: r.fun)

    gof = -best.fun
    theta_hat = best.x[0]
    se = np.sqrt(np.diag(best.hess_inv))[0]
    ci95_lower = theta_hat - 1.96 * se
    ci95_upper = theta_hat + 1.96 * se

    reference_ll = -neg_log_likelihood([0.0], demands, success, k)
    pseudo_r2 = 1 - (gof / reference_ll)

    result = {
        "theta_hat": theta_hat,
        "se": se,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "convergence": float(best.success),
        "reference_ll": reference_ll,
        "gof": gof,
        "pseudo_r2": pseudo_r2,
    }
    if n_restarts > 1:
        result["n_restarts"] = len(attempts)
        result["n_converged"] = len(converged)
        result["restart_theta_std"] = float(np.std([r.x[0] for r in attempts]))
    return result
