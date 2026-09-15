"""Phase 7: adversarial and correctness tests for
`treasury_auction_stress.features.safe_as_of_lookback.safe_as_of_trailing_mean`
-- the shared primitive behind both the adaptive recent-history
benchmark and the stress-event gate's safe regime feature. Covers
several of the Step 4 required adversarial cases directly: same-day/
same-tenor auctions, a neighboring auction whose result is not yet
available, publication-date equality at the cutoff, missing-value
rows, shuffled input rows, and future-row poisoning.
"""

from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.features.safe_as_of_lookback import (
    safe_as_of_trailing_mean,
)


def _row(key, tenor, auction_date, cutoff, safe_date, value):
    return {
        "auction_key": key,
        "tenor": tenor,
        "auction_date": pd.Timestamp(auction_date),
        "cutoff_date": pd.Timestamp(cutoff),
        "safe_date": pd.Timestamp(safe_date),
        "primary_dealer_share": value,
    }


def _compute(df: pd.DataFrame, *, n_lookback: int = 8) -> pd.DataFrame:
    return safe_as_of_trailing_mean(
        df,
        group_col="tenor",
        value_col="primary_dealer_share",
        key_col="auction_key",
        cutoff_col="cutoff_date",
        safe_date_col="safe_date",
        date_col="auction_date",
        n_lookback=n_lookback,
    )


def test_basic_trailing_mean_uses_only_eligible_predecessors():
    rows = [
        _row("a1", "2-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.10),
        _row("a2", "2-Year", "2020-01-12", "2020-01-12", "2020-01-13", 0.20),
        _row("a3", "2-Year", "2020-01-19", "2020-01-19", "2020-01-20", 0.30),
    ]
    df = pd.DataFrame(rows)
    out = _compute(df)

    # a1 has no eligible predecessor.
    a1 = out.loc[out["auction_key"] == "a1"].iloc[0]
    assert a1["primary_dealer_share_safe_trailing_n_used"] == 0
    assert pd.isna(a1["primary_dealer_share_safe_trailing_mean"])

    # a2's cutoff (2020-01-12) is after a1's safe_date (2020-01-06) -> eligible.
    a2 = out.loc[out["auction_key"] == "a2"].iloc[0]
    assert a2["primary_dealer_share_safe_trailing_n_used"] == 1
    assert a2["primary_dealer_share_safe_trailing_mean"] == pytest.approx(0.10)

    # a3's cutoff (2020-01-19) is after both a1 and a2's safe dates.
    a3 = out.loc[out["auction_key"] == "a3"].iloc[0]
    assert a3["primary_dealer_share_safe_trailing_n_used"] == 2
    assert a3["primary_dealer_share_safe_trailing_mean"] == pytest.approx(0.15)


def test_same_day_announcement_never_borrows_the_others_result():
    """Two same-tenor auctions announced (and cutoff-dated) on the exact
    same day: neither can see the other's result, regardless of which
    one appears first in the input frame.
    """
    rows = [
        _row("same_day_1", "7-Year", "2019-03-27", "2019-03-22", "2019-03-28", 0.40),
        _row("same_day_2", "7-Year", "2019-03-28", "2019-03-22", "2019-03-29", 0.90),
    ]
    df = pd.DataFrame(rows)
    out = _compute(df)
    for key in ("same_day_1", "same_day_2"):
        row = out.loc[out["auction_key"] == key].iloc[0]
        assert row["primary_dealer_share_safe_trailing_n_used"] == 0
        assert pd.isna(row["primary_dealer_share_safe_trailing_mean"])


def test_neighboring_auction_result_not_yet_available_is_excluded():
    """A same-tenor auction held a few days earlier, whose result is not
    yet safely available by this row's own cutoff, must be excluded --
    the real 2013-08-22 scenario documented in
    `dealer_absorption_audit.py`'s module docstring.
    """
    earlier = _row("earlier", "7-Year", "2013-08-28", "2013-08-22", "2013-08-29", 0.50)
    later = _row("later", "7-Year", "2013-08-29", "2013-08-22", "2013-08-30", 0.99)
    df = pd.DataFrame([earlier, later])
    out = _compute(df)
    later_row = out.loc[out["auction_key"] == "later"].iloc[0]
    # `later`'s own cutoff (2013-08-22) is BEFORE `earlier`'s safe_date
    # (2013-08-29) -- `earlier` must not be used.
    assert later_row["primary_dealer_share_safe_trailing_n_used"] == 0
    assert pd.isna(later_row["primary_dealer_share_safe_trailing_mean"])


def test_publication_date_exactly_equal_to_cutoff_is_eligible():
    """The rule is `safe_date <= cutoff` -- exact equality counts as
    eligible (a strict boundary case, not left ambiguous).
    """
    earlier = _row("earlier", "10-Year", "2021-06-01", "2021-06-01", "2021-06-08", 0.25)
    later = _row("later", "10-Year", "2021-06-15", "2021-06-08", "2021-06-16", 0.75)
    df = pd.DataFrame([earlier, later])
    out = _compute(df)
    later_row = out.loc[out["auction_key"] == "later"].iloc[0]
    assert later_row["primary_dealer_share_safe_trailing_n_used"] == 1
    assert later_row["primary_dealer_share_safe_trailing_mean"] == pytest.approx(0.25)


def test_missing_value_row_is_never_used_but_still_gets_its_own_output_row():
    rows = [
        _row("a1", "3-Year", "2018-01-01", "2018-01-01", "2018-01-02", float("nan")),
        _row("a2", "3-Year", "2018-02-01", "2018-02-01", "2018-02-02", 0.40),
        _row("a3", "3-Year", "2018-03-01", "2018-03-01", "2018-03-02", 0.60),
    ]
    df = pd.DataFrame(rows)
    out = _compute(df)

    a1_out = out.loc[out["auction_key"] == "a1"].iloc[0]
    assert a1_out["primary_dealer_share_safe_trailing_n_used"] == 0

    a3_out = out.loc[out["auction_key"] == "a3"].iloc[0]
    # a1 is missing and must not count as a used observation for a3.
    assert a3_out["primary_dealer_share_safe_trailing_n_used"] == 1
    assert a3_out["primary_dealer_share_safe_trailing_mean"] == pytest.approx(0.40)


def test_sparse_history_tenor_falls_back_to_fewer_than_n_lookback():
    rows = [
        _row("a1", "20-Year", "2020-06-01", "2020-06-01", "2020-06-02", 0.80),
        _row("a2", "20-Year", "2020-07-01", "2020-07-01", "2020-07-02", 0.70),
        _row("a3", "20-Year", "2020-08-01", "2020-08-01", "2020-08-02", 0.60),
    ]
    df = pd.DataFrame(rows)
    out = _compute(df, n_lookback=8)
    a3_out = out.loc[out["auction_key"] == "a3"].iloc[0]
    assert a3_out["primary_dealer_share_safe_trailing_n_used"] == 2
    assert a3_out["primary_dealer_share_safe_trailing_mean"] == pytest.approx(0.75)


def test_lookback_window_caps_at_n_lookback_most_recent_eligible():
    rows = [
        _row(f"a{i}", "5-Year", f"2020-01-{i:02d}", f"2020-01-{i:02d}", f"2020-01-{i + 1:02d}", float(i))
        for i in range(1, 11)
    ]
    df = pd.DataFrame(rows)
    out = _compute(df, n_lookback=3)
    a10 = out.loc[out["auction_key"] == "a10"].iloc[0]
    # a10's cutoff is 2020-01-10; a1..a9's safe_dates are 2020-01-02..2020-01-10,
    # so a9 (safe_date 2020-01-10) is exactly eligible; the 3 most
    # recent eligible are a7, a8, a9 (values 7, 8, 9).
    assert a10["primary_dealer_share_safe_trailing_n_used"] == 3
    assert a10["primary_dealer_share_safe_trailing_mean"] == pytest.approx(8.0)


def test_row_order_invariant_under_shuffling():
    rows = [
        _row("a1", "2-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.10),
        _row("a2", "2-Year", "2020-01-12", "2020-01-12", "2020-01-13", 0.20),
        _row("a3", "2-Year", "2020-01-19", "2020-01-19", "2020-01-20", 0.30),
        _row("b1", "30-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.90),
    ]
    baseline_order = pd.DataFrame(rows)
    shuffled_order = pd.DataFrame([rows[3], rows[1], rows[0], rows[2]])

    out_baseline = _compute(baseline_order).set_index("auction_key").sort_index()
    out_shuffled = _compute(shuffled_order).set_index("auction_key").sort_index()

    pd.testing.assert_series_equal(
        out_baseline["primary_dealer_share_safe_trailing_mean"],
        out_shuffled["primary_dealer_share_safe_trailing_mean"],
    )
    pd.testing.assert_series_equal(
        out_baseline["primary_dealer_share_safe_trailing_n_used"],
        out_shuffled["primary_dealer_share_safe_trailing_n_used"],
    )


def test_future_row_poisoning_never_changes_an_earlier_rows_output():
    rows = [
        _row("a1", "2-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.10),
        _row("a2", "2-Year", "2020-01-12", "2020-01-12", "2020-01-13", 0.20),
    ]
    without_future = pd.DataFrame(rows)
    with_future_poison = pd.DataFrame(
        rows + [_row("a3_future", "2-Year", "2030-01-01", "2030-01-01", "2030-01-02", 999.0)]
    )

    out_without = _compute(without_future)
    out_with = _compute(with_future_poison)

    a2_without = out_without.loc[out_without["auction_key"] == "a2"].iloc[0]
    a2_with = out_with.loc[out_with["auction_key"] == "a2"].iloc[0]
    assert a2_without["primary_dealer_share_safe_trailing_mean"] == pytest.approx(
        a2_with["primary_dealer_share_safe_trailing_mean"]
    )
    assert a2_without["primary_dealer_share_safe_trailing_n_used"] == a2_with["primary_dealer_share_safe_trailing_n_used"]


def test_year_boundary_december_to_january_is_not_special_cased():
    """A same-tenor auction just before a calendar year boundary must
    still be excluded if its own result is not yet safely available --
    the year boundary itself carries no special meaning to this
    function (only the safe-availability timestamp does).
    """
    dec_auction = _row("dec", "10-Year", "2019-12-30", "2019-12-24", "2019-12-31", 0.55)
    jan_auction = _row("jan", "10-Year", "2020-01-06", "2019-12-24", "2020-01-07", 0.65)
    df = pd.DataFrame([dec_auction, jan_auction])
    out = _compute(df)
    jan_row = out.loc[out["auction_key"] == "jan"].iloc[0]
    # jan's own cutoff (2019-12-24) is before dec's safe_date (2019-12-31).
    assert jan_row["primary_dealer_share_safe_trailing_n_used"] == 0


def test_raises_if_output_columns_already_present():
    rows = [_row("only", "2-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.10)]
    df = pd.DataFrame(rows)
    already_computed = _compute(df)
    with pytest.raises(ValueError, match="already present"):
        _compute(already_computed)


def test_self_never_counted_even_if_defensively_checked():
    rows = [_row("only", "2-Year", "2020-01-05", "2020-01-05", "2020-01-06", 0.10)]
    df = pd.DataFrame(rows)
    out = _compute(df)
    only_row = out.iloc[0]
    assert only_row["primary_dealer_share_safe_trailing_n_used"] == 0
    assert pd.isna(only_row["primary_dealer_share_safe_trailing_mean"])
