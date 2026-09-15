from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.features.treasury_rates_candidate_features import (
    add_curve_slopes_and_curvature,
    add_rate_changes_and_volatility,
    add_tenor_matched_rate_features,
    build_rate_feature_table,
)


def _wide(n_days: int = 30, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rate_date": dates,
            "publication_date": dates,
            "publication_safe_available_date": dates + pd.Timedelta(days=1),
            "2 Yr": 4.30 + rng.normal(0, 0.02, n_days).cumsum(),
            "10 Yr": 4.00 + rng.normal(0, 0.02, n_days).cumsum(),
            "5 Yr": 4.10 + rng.normal(0, 0.02, n_days).cumsum(),
            "30 Yr": 4.50 + rng.normal(0, 0.02, n_days).cumsum(),
        }
    )


def test_percentage_point_to_basis_point_conversion_is_exactly_100x():
    wide = pd.DataFrame(
        {
            "rate_date": pd.bdate_range("2024-01-02", periods=3),
            "2 Yr": [4.00, 4.01, 4.11],
        }
    )
    out = add_rate_changes_and_volatility(wide, change_windows=(1,), vol_windows=())
    # 4.01 - 4.00 = 0.01 percentage points = 1 basis point
    assert out.loc[1, "2 Yr_chg_1d_bps"] == pytest.approx(1.0)
    # 4.11 - 4.01 = 0.10 percentage points = 10 basis points
    assert out.loc[2, "2 Yr_chg_1d_bps"] == pytest.approx(10.0)


def test_one_percentage_point_equals_100_basis_points_directly():
    wide = pd.DataFrame({"rate_date": pd.bdate_range("2024-01-02", periods=2), "2 Yr": [4.00, 5.00]})
    out = add_rate_changes_and_volatility(wide, change_windows=(1,), vol_windows=())
    assert out.loc[1, "2 Yr_chg_1d_bps"] == pytest.approx(100.0)


def test_change_window_uses_correct_lag():
    wide = _wide()
    out = add_rate_changes_and_volatility(wide, change_windows=(5,), vol_windows=())
    row = 10
    expected = (wide.loc[row, "2 Yr"] - wide.loc[row - 5, "2 Yr"]) * 100
    assert out.loc[row, "2 Yr_chg_5d_bps"] == pytest.approx(expected)


def test_volatility_excludes_current_day_change():
    wide = _wide(n_days=40)
    out = add_rate_changes_and_volatility(wide, change_windows=(), vol_windows=(10,))
    row = 20
    daily_changes = wide["2 Yr"].diff() * 100
    prior_window = daily_changes.iloc[row - 10 : row]  # excludes row's own change (shift(1) semantics)
    expected = prior_window.std()
    assert out.loc[row, "2 Yr_vol_10d_bps"] == pytest.approx(expected)


def test_future_rows_do_not_change_earlier_volatility_values():
    full = _wide(n_days=40)
    truncated = full.iloc[:25].copy()
    out_full = add_rate_changes_and_volatility(full, change_windows=(), vol_windows=(10,))
    out_truncated = add_rate_changes_and_volatility(truncated, change_windows=(), vol_windows=(10,))
    pd.testing.assert_series_equal(
        out_full.loc[:24, "2 Yr_vol_10d_bps"].reset_index(drop=True),
        out_truncated["2 Yr_vol_10d_bps"].reset_index(drop=True),
    )


def test_insufficient_history_stays_missing_not_zero():
    wide = _wide(n_days=5)
    out = add_rate_changes_and_volatility(wide, change_windows=(20,), vol_windows=(20,))
    assert out["2 Yr_chg_20d_bps"].isna().all()
    assert out["2 Yr_vol_20d_bps"].isna().all()


def test_curve_slopes_and_curvature_arithmetic():
    wide = pd.DataFrame({"2 Yr": [4.00], "5 Yr": [4.10], "10 Yr": [4.30], "30 Yr": [4.60]})
    out = add_curve_slopes_and_curvature(wide)
    assert out.loc[0, "slope_2s10s_bps"] == pytest.approx((4.30 - 4.00) * 100)
    assert out.loc[0, "slope_5s30s_bps"] == pytest.approx((4.60 - 4.10) * 100)
    assert out.loc[0, "curvature_2_10_30_bps"] == pytest.approx((2 * 4.30 - 4.00 - 4.60) * 100)


def test_build_rate_feature_table_keeps_raw_levels():
    out = build_rate_feature_table(_wide())
    assert "2 Yr" in out.columns
    assert "2 Yr_chg_1d_bps" in out.columns
    assert "slope_2s10s_bps" in out.columns


def test_tenor_matched_features_select_correct_maturity_and_neighbors():
    joined = pd.DataFrame(
        {
            "tenor": ["7-Year", "2-Year"],
            "1 Mo": [4.0, 4.0],
            "6 Mo": [4.1, 4.1],
            "1 Yr": [4.2, 4.2],
            "2 Yr": [4.3, 4.3],
            "3 Yr": [4.4, 4.4],
            "5 Yr": [4.5, 4.5],
            "7 Yr": [4.6, 4.6],
            "10 Yr": [4.7, 4.7],
        }
    )
    out = add_tenor_matched_rate_features(joined)
    assert out.loc[0, "matched_tenor_par_yield_percent"] == 4.6  # 7-Year -> "7 Yr"
    assert out.loc[0, "adjacent_lower_maturity_label"] == "5 Yr"
    assert out.loc[0, "adjacent_higher_maturity_label"] == "10 Yr"
    assert out.loc[1, "matched_tenor_par_yield_percent"] == 4.3  # 2-Year -> "2 Yr"
    assert out.loc[1, "adjacent_lower_maturity_label"] == "1 Yr"
    assert out.loc[1, "adjacent_higher_maturity_label"] == "3 Yr"
