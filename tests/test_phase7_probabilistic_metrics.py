"""Phase 7: correctness tests for
`treasury_auction_stress.evaluation.probabilistic_metrics`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.probabilistic_metrics import (
    compute_coverage_table,
    compute_pinball_table,
    pinball_loss,
    summarize_probabilistic_by,
)


def test_pinball_loss_matches_hand_computed_values():
    actual = pd.Series([0.40, 0.40])
    forecast = pd.Series([0.30, 0.50])  # under-prediction, then over-prediction
    loss_q90 = pinball_loss(actual, forecast, 0.90)
    # under-prediction (diff=+0.10): loss = 0.90 * 0.10 = 0.09
    assert loss_q90.iloc[0] == pytest.approx(0.09)
    # over-prediction (diff=-0.10): loss = (0.90-1) * -0.10 = 0.01
    assert loss_q90.iloc[1] == pytest.approx(0.01)


def test_compute_pinball_table_scales_share_targets_to_percentage_points():
    df = pd.DataFrame({"actual": [0.40], "q0.50": [0.30]})
    result = compute_pinball_table(df, quantile_levels=(0.5,), target_name="primary_dealer_share")
    # median pinball loss = 0.5 * |0.40-0.30| = 0.05 (raw units) -> 5.0 pp
    assert result["pinball_q0.50"] == pytest.approx(5.0)
    assert result["pinball_mean"] == pytest.approx(5.0)
    assert result["unit"] == "percentage points"
    assert result["n"] == 1


def test_compute_pinball_table_empty_frame_returns_none_not_error():
    df = pd.DataFrame({"actual": [], "q0.50": []})
    result = compute_pinball_table(df, quantile_levels=(0.5,), target_name="primary_dealer_share")
    assert result["n"] == 0
    assert result["pinball_mean"] is None
    assert result["pinball_q0.50"] is None


def test_compute_coverage_table_hand_computed():
    df = pd.DataFrame(
        {
            "actual": [0.50, 0.05, 0.95],
            "q0.05": [0.10, 0.10, 0.10],
            "q0.95": [0.90, 0.90, 0.90],
            "q0.25": [0.40, 0.40, 0.40],
            "q0.75": [0.60, 0.60, 0.60],
            "q0.10": [0.20, 0.20, 0.20],
            "q0.90": [0.80, 0.80, 0.80],
        }
    )
    result = compute_coverage_table(df, target_name="primary_dealer_share")
    # 90% interval [0.10, 0.90]: row0 covered, row1 (0.05) NOT covered, row2 (0.95) NOT covered -> 1/3
    assert result["coverage_90%"] == pytest.approx(1 / 3)
    assert result["width_90%"] == pytest.approx((0.90 - 0.10) * 100.0)
    # 50% interval [0.40, 0.60]: only row0 (0.50) covered -> 1/3
    assert result["coverage_50%"] == pytest.approx(1 / 3)


def test_compute_coverage_table_empty_frame_returns_none():
    df = pd.DataFrame({"actual": [], "q0.05": [], "q0.95": [], "q0.25": [], "q0.75": [], "q0.10": [], "q0.90": []})
    result = compute_coverage_table(df, target_name="primary_dealer_share")
    assert result["n"] == 0
    assert result["coverage_90%"] is None


def test_summarize_probabilistic_by_reports_counts_per_group():
    df = pd.DataFrame(
        {
            "test_year": [2020, 2020, 2021],
            "actual": [0.40, 0.45, 0.30],
            "q0.50": [0.35, 0.40, 0.30],
            "q0.05": [0.10, 0.10, 0.10],
            "q0.95": [0.90, 0.90, 0.90],
            "q0.25": [0.20, 0.20, 0.20],
            "q0.75": [0.60, 0.60, 0.60],
            "q0.10": [0.15, 0.15, 0.15],
            "q0.90": [0.80, 0.80, 0.80],
        }
    )
    table = summarize_probabilistic_by(
        df, quantile_levels=(0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95), target_name="primary_dealer_share", group_col="test_year"
    )
    assert set(table["test_year"]) == {2020, 2021}
    assert table.loc[table["test_year"] == 2020, "n"].iloc[0] == 2
    assert table.loc[table["test_year"] == 2021, "n"].iloc[0] == 1
