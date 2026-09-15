"""Phase 7: correctness tests for `AdaptiveRecentHistoryBaseline`."""

from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.timing import RESULT_SAFE_AVAILABLE_DATE_COL
from treasury_auction_stress.models.adaptive_baselines import (
    AdaptiveRecentHistoryBaseline,
)

CUTOFF_COL = "announcement_cutoff_date"


def _row(key, tenor, auction_date, cutoff, safe_date, value):
    return {
        "auction_key": key,
        "tenor": tenor,
        "auction_date": pd.Timestamp(auction_date),
        CUTOFF_COL: pd.Timestamp(cutoff),
        RESULT_SAFE_AVAILABLE_DATE_COL: pd.Timestamp(safe_date),
        "primary_dealer_share": value,
    }


def test_adaptive_baseline_updates_within_the_test_year():
    train = pd.DataFrame(
        [
            _row("t1", "2-Year", "2014-01-01", "2014-01-01", "2014-01-02", 0.50),
            _row("t2", "2-Year", "2014-02-01", "2014-02-01", "2014-02-02", 0.50),
        ]
    )
    test = pd.DataFrame(
        [
            # early: sees only the 2 training rows -> mean 0.50
            _row("s1", "2-Year", "2015-01-05", "2015-01-05", "2015-01-06", 0.90),
            # late: s1's own result IS now safely available (safe_date 2015-01-06 <= this cutoff 2015-06-01)
            _row("s2", "2-Year", "2015-06-01", "2015-06-01", "2015-06-02", 0.10),
        ]
    )
    model = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(train)
    forecast = model.predict(test, cutoff_col=CUTOFF_COL)

    assert forecast.loc[test.index[0]] == pytest.approx(0.50)  # (0.50 + 0.50) / 2
    # s2 should now see t1, t2, AND s1 (0.90) -> mean (0.5+0.5+0.9)/3
    assert forecast.loc[test.index[1]] == pytest.approx((0.50 + 0.50 + 0.90) / 3)


def test_adaptive_baseline_matches_frozen_style_result_when_nothing_resolves_within_year():
    """If no test-year auction's result becomes safely available before
    another test-year auction's own cutoff (the ordinary case for
    Phase 6's frozen baseline), the adaptive model's prediction for
    every test row should equal the plain mean of the n_lookback most
    recent TRAINING rows -- identical to what a frozen baseline would
    produce, since nothing new became available within the year.
    """
    train = pd.DataFrame(
        [_row(f"t{i}", "5-Year", f"2013-0{i}-01", f"2013-0{i}-01", f"2013-0{i}-02", 0.10 * i) for i in range(1, 5)]
    )
    test = pd.DataFrame(
        [
            _row("s1", "5-Year", "2014-01-05", "2014-01-05", "2014-01-06", 0.99),
            _row("s2", "5-Year", "2014-01-06", "2014-01-05", "2014-01-07", 0.99),
        ]
    )
    model = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(train)
    forecast = model.predict(test, cutoff_col=CUTOFF_COL)
    expected = sum(0.10 * i for i in range(1, 5)) / 4
    assert forecast.loc[test.index[0]] == pytest.approx(expected)
    assert forecast.loc[test.index[1]] == pytest.approx(expected)


def test_adaptive_baseline_falls_back_to_global_mean_for_unseen_tenor():
    train = pd.DataFrame([_row("t1", "2-Year", "2014-01-01", "2014-01-01", "2014-01-02", 0.40)])
    test = pd.DataFrame([_row("s1", "20-Year", "2020-06-01", "2020-06-01", "2020-06-02", 0.99)])
    model = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(train)
    forecast = model.predict(test, cutoff_col=CUTOFF_COL)
    assert forecast.loc[test.index[0]] == pytest.approx(0.40)  # global training mean


def test_adaptive_baseline_never_uses_a_same_day_test_year_sibling():
    train = pd.DataFrame([_row("t1", "7-Year", "2014-01-01", "2014-01-01", "2014-01-02", 0.30)])
    test = pd.DataFrame(
        [
            _row("s1", "7-Year", "2015-03-27", "2015-03-22", "2015-03-28", 0.40),
            _row("s2", "7-Year", "2015-03-28", "2015-03-22", "2015-03-29", 0.90),
        ]
    )
    model = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(train)
    forecast = model.predict(test, cutoff_col=CUTOFF_COL)
    # s2's cutoff (2015-03-22) is before s1's safe_date (2015-03-28) --
    # s2 must fall back to the training-only mean, not see s1.
    assert forecast.loc[test.index[1]] == pytest.approx(0.30)


def test_raises_without_result_safe_available_date_column():
    train = pd.DataFrame([{"auction_key": "t1", "tenor": "2-Year", "auction_date": pd.Timestamp("2014-01-01"),
                           CUTOFF_COL: pd.Timestamp("2014-01-01"), "primary_dealer_share": 0.4}])
    test = pd.DataFrame([{"auction_key": "s1", "tenor": "2-Year", "auction_date": pd.Timestamp("2015-01-01"),
                          CUTOFF_COL: pd.Timestamp("2015-01-01"), "primary_dealer_share": 0.9}])
    model = AdaptiveRecentHistoryBaseline().fit(train)
    with pytest.raises(KeyError):
        model.predict(test, cutoff_col=CUTOFF_COL)


def test_candidate_pool_lets_a_year_boundary_straggler_become_usable_once_available():
    """Acceptance-review regression test: a prior-year auction whose own
    result becomes safely available AFTER the fold's own fit_origin
    (so it is excluded from `fold.train`, which the FROZEN baseline
    must still use) but BEFORE a later test-year row's own cutoff must
    be usable by the ADAPTIVE model once genuinely available -- when
    `candidate_pool` (the wider, non-fit_origin-limited prior-years
    pool) is supplied. Without `candidate_pool`, this straggler would
    be invisible for the whole test year (see the default-behavior
    test below), which is the narrower behavior this fix corrects.
    """
    straggler = _row("straggler_2014", "2-Year", "2014-12-20", "2014-12-20", "2015-06-01", 0.77)
    early_train = _row("early_2014", "2-Year", "2014-01-01", "2014-01-01", "2014-01-02", 0.40)
    fold_train = pd.DataFrame([early_train])  # simulates fold.train, which EXCLUDES the straggler
    wider_pool = pd.DataFrame([early_train, straggler])  # the wider, fit_origin-unrestricted pool

    test = pd.DataFrame(
        [
            _row("early_test", "2-Year", "2015-01-05", "2015-01-05", "2015-01-06", 0.10),
            _row("late_test", "2-Year", "2015-07-01", "2015-07-01", "2015-07-02", 0.90),
        ]
    )

    model_without_pool = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(fold_train)
    forecast_without_pool = model_without_pool.predict(test, cutoff_col=CUTOFF_COL)
    # Straggler invisible: late_test only sees early_2014 (0.40) and early_test (0.10).
    assert forecast_without_pool.loc[test.index[1]] == pytest.approx((0.40 + 0.10) / 2)

    model_with_pool = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(fold_train, candidate_pool=wider_pool)
    forecast_with_pool = model_with_pool.predict(test, cutoff_col=CUTOFF_COL)
    # Straggler now correctly included once available (safe_date 2015-06-01 <= late_test's cutoff 2015-07-01).
    assert forecast_with_pool.loc[test.index[1]] == pytest.approx((0.40 + 0.77 + 0.10) / 3)

    # early_test's own cutoff (2015-01-05) is BEFORE the straggler's
    # safe_date (2015-06-01) -- the straggler must still never leak
    # into an EARLIER test-year row, wider pool or not.
    assert forecast_with_pool.loc[test.index[0]] == pytest.approx(0.40)

    # global_mean_ (the fallback) is unaffected by candidate_pool -- it
    # is always computed from fold_train alone, never the wider pool.
    assert model_with_pool.global_mean_ == pytest.approx(model_without_pool.global_mean_)


def test_candidate_pool_defaults_to_train_df_when_omitted():
    """Backward-compatible: omitting `candidate_pool` must reproduce
    the exact pre-fix behavior (candidate_pool_ falls back to train_df).
    """
    train = pd.DataFrame(
        [_row("t1", "2-Year", "2014-01-01", "2014-01-01", "2014-01-02", 0.40)]
    )
    model = AdaptiveRecentHistoryBaseline(n_lookback=8).fit(train)
    pd.testing.assert_frame_equal(model.candidate_pool_.reset_index(drop=True), train.reset_index(drop=True))
