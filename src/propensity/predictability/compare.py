"""Side-by-side comparison of named feature-set configurations against the
same label, e.g. "does adding propensity columns to a capability-only
feature set improve an assessor's AUROC?"
"""

from typing import Dict, List

import pandas as pd

from .assessor import run_ten_fold_cv
from .features import build_feature_table


def compare_feature_sets(
    df: pd.DataFrame, feature_sets: Dict[str, List[str]], label_column: str, seed: int = 42
) -> pd.DataFrame:
    """Runs run_ten_fold_cv once per named feature set on the same
    underlying rows/label, returning a DataFrame with one row per set:
    columns "feature_set", "n_features", "n_rows", "auroc". When exactly
    two feature sets are given, an additional "delta_vs_<first_name>"
    column reports each row's AUROC minus the first set's -- the pairwise
    comparison this function exists for, generalized to N named sets since
    that costs nothing extra.
    """
    names = list(feature_sets.keys())
    rows = []
    for name, columns in feature_sets.items():
        X, y = build_feature_table(df, columns, label_column)
        auroc = run_ten_fold_cv(X, y, seed=seed)
        rows.append({"feature_set": name, "n_features": len(columns), "n_rows": len(X), "auroc": auroc})

    result = pd.DataFrame(rows, columns=["feature_set", "n_features", "n_rows", "auroc"])

    if len(names) == 2:
        baseline = result.loc[result["feature_set"] == names[0], "auroc"].iloc[0]
        result[f"delta_vs_{names[0]}"] = result["auroc"] - baseline

    return result
