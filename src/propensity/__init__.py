"""Derivation of propensity surfaces and curves (arXiv 2602.18182, stage 3).

- model: canonical Eq. 2-5 propensity model (two_sided_sigma).
- variants: exploratory alternatives from early development (reference only).
- data: loading demand annotations (JSONL) and outcomes (wide CSV).
- mle: Eq. 6 maximum-likelihood fitting of theta.
- curves: empirical propensity curve construction.
- surfaces: empirical propensity surface construction.
- plotting: matplotlib figures for both.
"""

from .model import two_sided_sigma
from .mle import fit_theta, neg_log_likelihood
from .data import load_demands, load_outcomes, build_arrays
from .curves import build_empirical_curve
from .surfaces import build_empirical_surface
from .plotting import plot_prop_curve, plot_prop_surface

__all__ = [
    "two_sided_sigma",
    "fit_theta",
    "neg_log_likelihood",
    "load_demands",
    "load_outcomes",
    "build_arrays",
    "build_empirical_curve",
    "build_empirical_surface",
    "plot_prop_curve",
    "plot_prop_surface",
]
