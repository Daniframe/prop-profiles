"""Batch theta-fitting across models x incitement levels, producing the same
table shape as prop-meas-anon's Derived_propensities_{dataset}.csv. Factors
get_all_prop_points.py's subprocess-per-(model,level) loop into an in-process
function calling mle.fit_theta directly.
"""

from typing import List, Optional, Union

import pandas as pd

from .data import build_arrays
from .mle import fit_theta

DEFAULT_LEVELS = ["-3", "-2", "-1", "0", "+1", "+2", "+3", None]


def derive_propensity_table(
    demands_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    models: List[str],
    code: str,
    levels: List[Optional[Union[str, int]]] = DEFAULT_LEVELS,
    k: float = 1.0,
    n_bins: int = 20,
    lowess_frac: float = 0.4,
    maxiter: int = 500,
    n_restarts: int = 1,
) -> pd.DataFrame:
    """Fits theta for every (model, level) pair, skipping any pair whose
    outcome column isn't present in outcomes_df (e.g. a model that wasn't
    run at a given level) or that has zero rows after the demands/outcomes
    join.

    levels: incitement levels to test; None means the "unprompted" column
        {model}_{code}_outcome (no level suffix) -- matches
        get_all_prop_points.py's LEVELS list, which included a bare "" entry.
    n_restarts: forwarded to mle.fit_theta -- 1 (default) is a single BFGS
        run per pair, unchanged from before this parameter existed; >1 retries
        from multiple starting points and keeps the best-likelihood converged
        result, which matters for near-ceiling/near-floor outcome data where
        a single run can land in an arbitrary local optimum (see fit_theta's
        docstring for why this keeps the best mode rather than averaging).

    Returns a DataFrame with columns Model, Incited propensity, Obtained
    propensity, Lower 95CI, Upper 95CI, Converges, ReferenceLL, EstimationLL
    -- the same schema as Derived_propensities_{dataset}.csv. When
    n_restarts > 1, also includes NRestarts/NConverged/RestartThetaStd.
    """
    rows = []
    for model in models:
        for level in levels:
            model_name = f"{model}_{code}" if level is None else f"{model}_{code}_{level}"
            if f"{model_name}_outcome" not in outcomes_df.columns:
                continue

            demands, success = build_arrays(demands_df, outcomes_df, model_name)
            if len(demands) == 0:
                continue

            fit = fit_theta(demands, success, k=k, n_bins=n_bins, lowess_frac=lowess_frac,
                             maxiter=maxiter, n_restarts=n_restarts)
            incited_prop = None if level is None else float(level)

            row = {
                "Model": model,
                "Incited propensity": incited_prop,
                "Obtained propensity": fit["theta_hat"],
                "Lower 95CI": fit["ci95_lower"],
                "Upper 95CI": fit["ci95_upper"],
                "Converges": fit["convergence"],
                "ReferenceLL": fit["reference_ll"],
                "EstimationLL": fit["gof"],
            }
            if n_restarts > 1:
                row["NRestarts"] = fit["n_restarts"]
                row["NConverged"] = fit["n_converged"]
                row["RestartThetaStd"] = fit["restart_theta_std"]
            rows.append(row)

    columns = ["Model", "Incited propensity", "Obtained propensity", "Lower 95CI",
               "Upper 95CI", "Converges", "ReferenceLL", "EstimationLL"]
    if n_restarts > 1:
        columns += ["NRestarts", "NConverged", "RestartThetaStd"]
    return pd.DataFrame(rows, columns=columns)
