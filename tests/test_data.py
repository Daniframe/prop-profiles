import json

import pandas as pd

from src.propensity.data import load_demands, load_outcomes, build_arrays


def test_build_arrays_joins_correctly(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    outcomes_path = tmp_path / "outcomes.csv"

    demand_rows = [
        {"question_id": 1, "propensity_lower": -2.0, "propensity_upper": 0.0},
        {"question_id": 2, "propensity_lower": -1.0, "propensity_upper": 1.0},
        {"question_id": 3, "propensity_lower": 0.0, "propensity_upper": 2.0},
    ]
    with open(demands_path, "w") as f:
        for row in demand_rows:
            f.write(json.dumps(row) + "\n")

    outcomes_df = pd.DataFrame({
        "question_id": [1, 2, 3],
        "model_a_outcome": [1, 0, 1],
        "model_b_outcome": [0, 1, 1],
    })
    outcomes_df.to_csv(outcomes_path, index=False)

    demands_df = load_demands(str(demands_path))
    outcomes_df_loaded = load_outcomes(str(outcomes_path))

    assert list(demands_df.columns) == ["question_id", "lower", "upper"]

    demands, success = build_arrays(demands_df, outcomes_df_loaded, "model_a")
    assert demands.shape == (3, 2)
    assert list(demands[:, 0]) == [-2.0, -1.0, 0.0]
    assert list(demands[:, 1]) == [0.0, 1.0, 2.0]
    assert list(success) == [1, 0, 1]

    _, success_b = build_arrays(demands_df, outcomes_df_loaded, "model_b")
    assert list(success_b) == [0, 1, 1]


def test_build_arrays_inner_join_drops_unmatched(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    outcomes_path = tmp_path / "outcomes.csv"

    with open(demands_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "propensity_lower": -1.0, "propensity_upper": 1.0}) + "\n")
        f.write(json.dumps({"question_id": 2, "propensity_lower": -2.0, "propensity_upper": 2.0}) + "\n")

    pd.DataFrame({"question_id": [1], "model_a_outcome": [1]}).to_csv(outcomes_path, index=False)

    demands_df = load_demands(str(demands_path))
    outcomes_df = load_outcomes(str(outcomes_path))
    demands, success = build_arrays(demands_df, outcomes_df, "model_a")

    assert demands.shape == (1, 2)
    assert list(success) == [1]
