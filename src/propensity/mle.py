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


def fit_theta(demands, success, k=1.0, x_init=None, n_bins=20, lowess_frac=0.4, maxiter=500):
    """
    Fits theta by maximum likelihood (Eq. 6) via scipy.optimize.minimize
    (BFGS). If x_init is not given, derives one from a LOWESS-smoothed,
    binned empirical success curve (bin by interval center, smooth, take the
    argmax) -- exactly the approach used in get_prop_point.py.

    demands: (N, 2) array-like of (b_l, b_u) pairs.
    success: (N,) array-like of binary outcomes.

    Returns a dict: theta_hat, se, ci95_lower, ci95_upper, convergence,
    reference_ll, gof, pseudo_r2.
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

    res = minimize(
        neg_log_likelihood,
        x0=x_init,
        args=(demands, success, k),
        method="BFGS",
        options={"maxiter": maxiter},
    )

    gof = -res.fun
    theta_hat = res.x[0]
    se = np.sqrt(np.diag(res.hess_inv))[0]
    ci95_lower = theta_hat - 1.96 * se
    ci95_upper = theta_hat + 1.96 * se

    reference_ll = -neg_log_likelihood([0.0], demands, success, k)
    pseudo_r2 = 1 - (gof / reference_ll)

    return {
        "theta_hat": theta_hat,
        "se": se,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "convergence": float(res.success),
        "reference_ll": reference_ll,
        "gof": gof,
        "pseudo_r2": pseudo_r2,
    }
