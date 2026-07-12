"""Sector 5: predictability -- does a feature set (capability demands,
propensity demands, or any other item-level columns) help predict
per-instance model correctness, and does adding one feature set to another
improve on it?

- features: build_feature_table, a generic column-selection + NaN-drop step.
- assessor: run_ten_fold_cv, the one evaluation method implemented so far.
- compare: compare_feature_sets, side-by-side AUROC across named feature sets.
"""

from .features import build_feature_table
from .assessor import run_ten_fold_cv
from .compare import compare_feature_sets

__all__ = [
    "build_feature_table",
    "run_ten_fold_cv",
    "compare_feature_sets",
]
