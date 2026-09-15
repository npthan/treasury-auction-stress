from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.features.auction_candidate_features import (
    add_announcement_to_auction_gap,
    add_calendar_features,
    add_offering_amount_features,
    add_previous_same_tenor_features,
    add_prior_reopening_count,
    add_trailing_supply_features,
    build_auction_structure_feature_table,
)


def _auctions(rows: list[dict]) -> pd.DataFrame:
    defaults = {"is_reopening": False, "offering_amt": 40_000_000_000.0}
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    df["announcemt_date"] = pd.to_datetime(df["announcemt_date"])
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    return df


def test_log_offering_amt_is_deterministic_log1p():
    df = _auctions([{"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 100.0}])
    out = add_offering_amount_features(df)
    assert out["log_offering_amt"].iloc[0] == np.log1p(100.0)


def test_days_announcement_to_auction():
    df = _auctions([{"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-08"}])
    out = add_announcement_to_auction_gap(df)
    assert out["days_announcement_to_auction"].iloc[0] == 7


def test_calendar_features_known_from_auction_date_itself():
    df = _auctions([{"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-03-04", "auction_date": "2024-03-11"}])
    out = add_calendar_features(df)
    assert out["auction_month"].iloc[0] == 3
    assert out["auction_quarter"].iloc[0] == 1
    assert out["auction_day_of_week"].iloc[0] == pd.Timestamp("2024-03-11").dayofweek
    assert out["announcement_day_of_week"].iloc[0] == pd.Timestamp("2024-03-04").dayofweek


def test_previous_same_tenor_features_uses_strictly_earlier_announcement():
    df = _auctions(
        [
            {"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 40e9},
            {"cusip": "B", "tenor": "2-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "offering_amt": 42e9},
        ]
    )
    out = add_previous_same_tenor_features(df)
    first, second = out.iloc[0], out.iloc[1]
    assert pd.isna(first["previous_same_tenor_offering_amt"])
    assert second["previous_same_tenor_offering_amt"] == 40e9
    assert second["change_in_offering_amt_vs_previous_same_tenor"] == 2e9
    assert second["days_since_prior_same_tenor_auction"] == (pd.Timestamp("2024-02-05") - pd.Timestamp("2024-01-05")).days


def test_previous_same_tenor_features_ignores_a_different_tenor():
    df = _auctions(
        [
            {"cusip": "A", "tenor": "10-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 40e9},
            {"cusip": "B", "tenor": "2-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "offering_amt": 42e9},
        ]
    )
    out = add_previous_same_tenor_features(df)
    assert pd.isna(out.iloc[1]["previous_same_tenor_offering_amt"])


def test_same_day_same_tenor_announcement_never_precedes_the_other():
    """The verified real-data scenario (see module docstring): a
    reopening and a new issue of the same tenor announced on the exact
    same calendar date. Neither may be treated as the other's
    "previous" auction.
    """
    df = _auctions(
        [
            {"cusip": "OLD", "tenor": "5-Year", "announcemt_date": "2024-05-01", "auction_date": "2024-05-06", "is_reopening": True, "offering_amt": 30e9},
            {"cusip": "NEW", "tenor": "5-Year", "announcemt_date": "2024-05-01", "auction_date": "2024-05-07", "is_reopening": False, "offering_amt": 45e9},
        ]
    )
    out = add_previous_same_tenor_features(df)
    assert out["previous_same_tenor_offering_amt"].isna().all()
    assert out["days_since_prior_same_tenor_auction"].isna().all()


def test_prior_reopening_count_counts_only_strictly_earlier_reopenings():
    df = _auctions(
        [
            {"cusip": "X", "tenor": "10-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "is_reopening": False},
            {"cusip": "X", "tenor": "10-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "is_reopening": True},
            {"cusip": "X", "tenor": "10-Year", "announcemt_date": "2024-03-01", "auction_date": "2024-03-05", "is_reopening": True},
        ]
    )
    out = add_prior_reopening_count(df)
    assert list(out["n_prior_reopenings_of_cusip"]) == [0, 0, 1]


def test_prior_reopening_count_same_day_reopenings_do_not_count_each_other():
    df = _auctions(
        [
            {"cusip": "X", "tenor": "10-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "is_reopening": True},
            {"cusip": "X", "tenor": "10-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-06", "is_reopening": True},
        ]
    )
    out = add_prior_reopening_count(df)
    assert list(out["n_prior_reopenings_of_cusip"]) == [0, 0]


def test_trailing_supply_excludes_current_row_and_same_day_batch():
    df = _auctions(
        [
            {"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 10e9},
            {"cusip": "B", "tenor": "10-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-06", "offering_amt": 20e9},
            {"cusip": "C", "tenor": "2-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "offering_amt": 15e9},
        ]
    )
    out = add_trailing_supply_features(df, all_tenor_windows=(2,), same_tenor_windows=(2,))
    row0, row1, row2 = out.iloc[0], out.iloc[1], out.iloc[2]
    # The two same-day (2024-01-01) auctions must not see each other's supply.
    assert pd.isna(row0["trailing_nominal_coupon_supply_2"])
    assert pd.isna(row1["trailing_nominal_coupon_supply_2"])
    # The third row (a later, distinct date) sees the pooled sum of the prior date's batch.
    assert row2["trailing_nominal_coupon_supply_2"] == 30e9
    # Same-tenor trailing: row2 (2-Year) sees only row0's own 2-Year supply, not row1's 10-Year.
    assert row2["trailing_same_tenor_supply_2"] == 10e9


def test_trailing_supply_never_includes_current_row_single_window():
    df = _auctions(
        [
            {"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 10e9},
        ]
    )
    out = add_trailing_supply_features(df, all_tenor_windows=(1,), same_tenor_windows=(1,))
    assert pd.isna(out["trailing_nominal_coupon_supply_1"].iloc[0])
    assert pd.isna(out["trailing_same_tenor_supply_1"].iloc[0])


def test_outcome_poisoning_invariance():
    """Replacing every auction-result field with extreme values must
    leave every auction-structure feature bit-identical -- none of them
    ever reads a result field.
    """
    df = _auctions(
        [
            {"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 10e9},
            {"cusip": "B", "tenor": "2-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "offering_amt": 12e9, "is_reopening": True},
        ]
    )
    baseline = build_auction_structure_feature_table(df)

    poisoned = df.copy()
    poisoned["high_yield"] = [999.0, -999.0]
    poisoned["primary_dealer_accepted"] = [1e15, -1e15]
    poisoned["bid_to_cover_ratio"] = [1e9, np.nan]
    poisoned_out = build_auction_structure_feature_table(poisoned)

    new_cols = [c for c in baseline.columns if c not in df.columns]
    pd.testing.assert_frame_equal(baseline[new_cols], poisoned_out[new_cols])


def test_input_order_invariance():
    df = _auctions(
        [
            {"cusip": "A", "tenor": "2-Year", "announcemt_date": "2024-01-01", "auction_date": "2024-01-05", "offering_amt": 10e9},
            {"cusip": "B", "tenor": "2-Year", "announcemt_date": "2024-02-01", "auction_date": "2024-02-05", "offering_amt": 12e9},
            {"cusip": "C", "tenor": "10-Year", "announcemt_date": "2024-01-15", "auction_date": "2024-01-20", "offering_amt": 30e9},
        ]
    )
    forward = build_auction_structure_feature_table(df).sort_values("cusip").reset_index(drop=True)
    shuffled_input = df.iloc[[2, 0, 1]].reset_index(drop=True)
    backward = build_auction_structure_feature_table(shuffled_input).sort_values("cusip").reset_index(drop=True)
    new_cols = [c for c in forward.columns if c not in df.columns]
    pd.testing.assert_frame_equal(forward[new_cols], backward[new_cols])
