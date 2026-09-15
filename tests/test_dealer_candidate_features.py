from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.features.dealer_candidate_features import (
    add_cross_maturity_imbalance,
    add_harmonized_positions,
    add_inventory_relative_to_offering,
    add_inventory_relative_to_offering_harmonized,
    add_rolling_zscore,
    add_trailing_changes,
    add_weekly_change,
    build_dealer_feature_table,
)


def _wide(n_weeks: int = 10, start: str = "2026-01-07") -> pd.DataFrame:
    obs = pd.date_range(start, periods=n_weeks, freq="7D")
    return pd.DataFrame(
        {
            "observation_date": obs,
            "publication_date": obs + pd.Timedelta(days=1),
            "publication_safe_available_date": obs + pd.Timedelta(days=2),
            "dealer_net_position_bills": np.arange(n_weeks, dtype="float64") * 10.0,
            "dealer_net_position_coupons_le_2y": np.arange(n_weeks, dtype="float64") * 5.0,
            "dealer_net_position_coupons_11y_21y": [np.nan] * 3 + list(np.arange(n_weeks - 3, dtype="float64")),
            "dealer_net_position_coupons_gt_21y": [np.nan] * 3 + list(np.arange(n_weeks - 3, dtype="float64") * 2),
        }
    )


def test_weekly_change_is_diff_from_immediately_prior_week():
    out = add_weekly_change(_wide(), base_columns=["dealer_net_position_bills"])
    assert pd.isna(out.loc[0, "dealer_net_position_bills_wow_change"])
    assert out.loc[1, "dealer_net_position_bills_wow_change"] == 10.0
    assert out.loc[5, "dealer_net_position_bills_wow_change"] == 10.0


def test_trailing_changes_use_correct_lag():
    out = add_trailing_changes(_wide(), base_columns=["dealer_net_position_bills"], windows=(4,))
    assert pd.isna(out.loc[3, "dealer_net_position_bills_chg_4w"])
    assert out.loc[4, "dealer_net_position_bills_chg_4w"] == 40.0  # 40 - 0


def test_rolling_zscore_excludes_current_row():
    wide = _wide(n_weeks=20)
    out = add_rolling_zscore(wide, base_columns=["dealer_net_position_bills"], window=8, min_periods=3)
    row = 10
    prior_window = wide["dealer_net_position_bills"].iloc[row - 8 : row]
    expected = (wide["dealer_net_position_bills"].iloc[row] - prior_window.mean()) / prior_window.std()
    assert out.loc[row, "dealer_net_position_bills_zscore_8w"] == pytest.approx(expected)


def test_rolling_zscore_is_missing_before_min_periods():
    out = add_rolling_zscore(_wide(n_weeks=10), base_columns=["dealer_net_position_bills"], window=8, min_periods=5)
    assert pd.isna(out.loc[0, "dealer_net_position_bills_zscore_8w"])
    assert pd.isna(out.loc[4, "dealer_net_position_bills_zscore_8w"])  # only 4 prior weeks exist


def test_future_rows_do_not_change_earlier_rolling_zscore_values():
    full = _wide(n_weeks=30)
    truncated = full.iloc[:15].copy()

    out_full = add_rolling_zscore(full, base_columns=["dealer_net_position_bills"], window=8, min_periods=3)
    out_truncated = add_rolling_zscore(truncated, base_columns=["dealer_net_position_bills"], window=8, min_periods=3)

    pd.testing.assert_series_equal(
        out_full.loc[:14, "dealer_net_position_bills_zscore_8w"].reset_index(drop=True),
        out_truncated["dealer_net_position_bills_zscore_8w"].reset_index(drop=True),
    )


def test_cross_maturity_imbalance_is_missing_before_long_bucket_regime_starts():
    wide = _wide()
    out = add_cross_maturity_imbalance(wide)
    assert pd.isna(out.loc[0, "dealer_long_short_imbalance"])  # long-end buckets are NaN here
    assert pd.isna(out.loc[2, "dealer_long_short_imbalance"])
    assert not pd.isna(out.loc[3, "dealer_long_short_imbalance"])


def test_cross_maturity_imbalance_arithmetic():
    wide = _wide()
    out = add_cross_maturity_imbalance(wide)
    row = 5
    expected = (
        wide.loc[row, "dealer_net_position_coupons_11y_21y"] + wide.loc[row, "dealer_net_position_coupons_gt_21y"]
    ) - (wide.loc[row, "dealer_net_position_bills"] + wide.loc[row, "dealer_net_position_coupons_le_2y"])
    assert out.loc[row, "dealer_long_short_imbalance"] == pytest.approx(expected)


def test_cross_maturity_imbalance_raises_on_missing_columns():
    with pytest.raises(KeyError):
        add_cross_maturity_imbalance(pd.DataFrame({"observation_date": [pd.Timestamp("2026-01-01")]}))


def test_build_dealer_feature_table_keeps_raw_levels_alongside_derived_columns():
    out = build_dealer_feature_table(_wide())
    assert "dealer_net_position_bills" in out.columns  # raw level retained
    assert "dealer_net_position_bills_wow_change" in out.columns
    assert "dealer_net_position_bills_chg_4w" in out.columns
    assert any(c.endswith("_zscore_52w") for c in out.columns)
    assert "dealer_long_short_imbalance" in out.columns
    assert "dealer_long_short_imbalance_zscore_52w" in out.columns


def test_inventory_relative_to_offering_converts_millions_to_dollars_correctly():
    joined = pd.DataFrame(
        {
            "tenor": ["2-Year", "10-Year"],
            "offering_amt": [50_000_000_000.0, 40_000_000_000.0],
            "dealer_net_position_coupons_le_2y": [100.0, np.nan],  # $100 million
            "dealer_net_position_coupons_7y_11y": [np.nan, 200.0],  # $200 million
        }
    )
    out = add_inventory_relative_to_offering(joined)
    assert out.loc[0, "dealer_tenor_bucket_inventory_millions"] == 100.0
    assert out.loc[0, "dealer_inventory_to_offering_ratio"] == pytest.approx(
        (100.0 * 1_000_000) / 50_000_000_000.0
    )
    assert out.loc[1, "dealer_inventory_to_offering_ratio"] == pytest.approx(
        (200.0 * 1_000_000) / 40_000_000_000.0
    )


def _harmonization_wide(n_weeks: int = 1150, start: str = "2001-07-04") -> pd.DataFrame:
    obs = pd.date_range(start, periods=n_weeks, freq="7D")
    rng = np.random.default_rng(1)
    legacy_end = pd.Timestamp("2013-03-27")
    middle_end = pd.Timestamp("2021-12-29")
    long_split = pd.Timestamp("2022-01-05")
    modern_start = pd.Timestamp("2013-04-03")
    return pd.DataFrame(
        {
            "observation_date": obs,
            "dealer_net_position_bills": rng.normal(40_000, 1000, n_weeks),
            "dealer_net_position_coupons_3y_6y": rng.normal(15_000, 500, n_weeks),
            "dealer_net_position_coupons_le_2y": np.where(obs >= modern_start, rng.normal(20_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_2y_3y": np.where(obs >= modern_start, rng.normal(10_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_6y_7y": np.where(obs >= modern_start, rng.normal(8_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_7y_11y": np.where(obs >= modern_start, rng.normal(12_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_11y_21y": np.where(obs >= long_split, rng.normal(15_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_gt_21y": np.where(obs >= long_split, rng.normal(30_000, 500, n_weeks), np.nan),
            "dealer_net_position_total_ex_tips": np.where(obs >= modern_start, rng.normal(150_000, 2000, n_weeks), np.nan),
            "dealer_net_position_coupons_le_3y_legacy_component": np.where(obs <= legacy_end, rng.normal(25_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_6y_11y_legacy_component": np.where(obs <= legacy_end, rng.normal(18_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_gt_11y_legacy_component": np.where(obs <= legacy_end, rng.normal(12_000, 500, n_weeks), np.nan),
            "dealer_net_position_coupons_gt_11y_discontinued_component": np.where(
                (obs > legacy_end) & (obs <= middle_end), rng.normal(40_000, 500, n_weeks), np.nan
            ),
        }
    )


def test_add_harmonized_positions_total_reconciles_exactly_in_legacy_window():
    wide = _harmonization_wide()
    out = add_harmonized_positions(wide)
    row = out.loc[out["observation_date"] == pd.Timestamp("2010-06-16")].iloc[0]
    manual = (
        row["dealer_net_position_bills"]
        + row["dealer_net_position_coupons_3y_6y"]
        + row["dealer_net_position_coupons_le_3y_legacy_component"]
        + row["dealer_net_position_coupons_6y_11y_legacy_component"]
        + row["dealer_net_position_coupons_gt_11y_legacy_component"]
    )
    assert row["dealer_net_position_total_ex_tips_harmonized"] == pytest.approx(manual)


def test_add_harmonized_positions_total_matches_modern_total_after_2013():
    wide = _harmonization_wide()
    out = add_harmonized_positions(wide)
    row = out.loc[out["observation_date"] == pd.Timestamp("2020-06-17")].iloc[0]
    assert row["dealer_net_position_total_ex_tips_harmonized"] == pytest.approx(
        row["dealer_net_position_total_ex_tips"]
    )


def test_add_harmonized_positions_gt_11y_continuous_across_all_three_regimes():
    wide = _harmonization_wide()
    out = add_harmonized_positions(wide)
    for date in ("2010-06-16", "2018-06-13", "2023-06-14"):
        value = out.loc[out["observation_date"] == pd.Timestamp(date), "dealer_net_position_coupons_gt_11y_harmonized"]
        assert value.notna().all(), f"expected a harmonized value on {date}"


def test_add_harmonized_positions_raises_on_overlapping_sources():
    wide = _harmonization_wide()
    # Corrupt the data so the legacy component and the modern sum both
    # have a real value for the same week -- an impossible, corrupted
    # input this function must refuse to silently resolve.
    corrupted = wide.copy()
    modern_start_idx = corrupted.index[corrupted["observation_date"] == pd.Timestamp("2013-04-03")][0]
    corrupted.loc[modern_start_idx, "dealer_net_position_coupons_le_3y_legacy_component"] = 99999.0
    with pytest.raises(ValueError):
        add_harmonized_positions(corrupted)


def test_add_inventory_relative_to_offering_harmonized_converts_correctly():
    joined = pd.DataFrame(
        {
            "tenor": ["2-Year", "20-Year"],
            "offering_amt": [50_000_000_000.0, 20_000_000_000.0],
            "dealer_net_position_coupons_le_3y_harmonized": [300.0, np.nan],
            "dealer_net_position_coupons_gt_11y_harmonized": [np.nan, 400.0],
        }
    )
    out = add_inventory_relative_to_offering_harmonized(joined)
    assert out.loc[0, "dealer_tenor_bucket_inventory_millions_harmonized"] == 300.0
    assert out.loc[0, "dealer_inventory_to_offering_ratio_harmonized"] == pytest.approx(
        (300.0 * 1_000_000) / 50_000_000_000.0
    )
    assert out.loc[1, "dealer_inventory_to_offering_ratio_harmonized"] == pytest.approx(
        (400.0 * 1_000_000) / 20_000_000_000.0
    )
