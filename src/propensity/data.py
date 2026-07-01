"""I/O matching the annotation/inference stages' actual output formats:
demand annotations as JSONL, model outcomes as a wide-format CSV (one
`{model_name}_outcome` column per model). Ported from get_prop_point.py.
"""

import pandas as pd

def load_demands(jsonl_path):
    """Reads a JSONL file with columns question_id, propensity_lower,
    propensity_upper; renames the latter two to lower/upper."""
    return pd.read_json(jsonl_path, lines=True).rename(
        columns={"propensity_upper": "upper", "propensity_lower": "lower"}
    )

def load_outcomes(csv_path):
    """Reads a wide-format outcomes CSV: question_id + one
    `{model_name}_outcome` column per model."""
    return pd.read_csv(csv_path)

def build_arrays(demands_df, outcomes_df, model_name):
    """Inner-joins demands and outcomes on question_id and extracts the
    (b_l, b_u) demand array and the binary success array for one model.

    Returns (demands: (N, 2) ndarray of [lower, upper], success: (N,) ndarray).
    """
    df = pd.merge(demands_df, outcomes_df, on="question_id", how="inner")
    demands = df[["lower", "upper"]].values
    success = df[f"{model_name}_outcome"].values
    return demands, success
