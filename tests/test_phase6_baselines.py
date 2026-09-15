from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.models.baselines import (
    GlobalMeanBaseline,
    RecentHistoryBaseline,
    TenorMeanBaseline,
    TenorReopeningMeanBaseline,
)


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": ["2-Year"] * 4 + ["10-Year"] * 4,
            "is_reopening": [False, False, True, True, False, False, False, True],
            "auction_date": pd.to_datetime(
                ["2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01", "2010-01-15", "2010-02-15", "2010-03-15", "2010-04-15"]
            ),
            "auction_key": [f"K{i}" for i in range(8)],
            "primary_dealer_share": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        }
    )


def test_global_mean_baseline():
    model = GlobalMeanBaseline().fit(_train_df())
    test_df = pd.DataFrame({"tenor": ["2-Year", "30-Year"]})
    preds = model.predict(test_df)
    assert preds.iloc[0] == pytest.approx(0.45)
    assert preds.iloc[1] == pytest.approx(0.45)


def test_global_mean_baseline_raises_on_all_missing_target():
    df = _train_df()
    df["primary_dealer_share"] = pd.NA
    with pytest.raises(ValueError):
        GlobalMeanBaseline().fit(df)


def test_tenor_mean_baseline_and_fallback_for_unseen_tenor():
    model = TenorMeanBaseline().fit(_train_df())
    test_df = pd.DataFrame({"tenor": ["2-Year", "10-Year", "20-Year"]})
    preds = model.predict(test_df)
    assert preds.iloc[0] == pytest.approx(0.25)
    assert preds.iloc[1] == pytest.approx(0.65)
    assert preds.iloc[2] == pytest.approx(model.global_mean_)  # unseen 20-Year falls back


def test_tenor_reopening_mean_fallback_hierarchy():
    model = TenorReopeningMeanBaseline().fit(_train_df())
    test_df = pd.DataFrame(
        {
            "tenor": ["2-Year", "2-Year", "10-Year", "30-Year"],
            "is_reopening": [False, True, False, False],
        }
    )
    preds = model.predict(test_df)
    assert preds.iloc[0] == pytest.approx((0.10 + 0.20) / 2)  # exact group mean
    assert preds.iloc[1] == pytest.approx((0.30 + 0.40) / 2)  # exact group mean
    # 10-Year, is_reopening=False group exists (3 rows) -> exact group mean
    assert preds.iloc[2] == pytest.approx((0.50 + 0.60 + 0.70) / 3)
    # 30-Year never seen at all -> falls all the way back to global mean
    assert preds.iloc[3] == pytest.approx(model.global_mean_)


def test_tenor_reopening_mean_falls_back_to_tenor_mean_for_unseen_group_combo():
    train = pd.DataFrame(
        {
            "tenor": ["10-Year", "10-Year", "10-Year"],
            "is_reopening": [False, False, False],  # no is_reopening=True row at all
            "auction_date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
            "primary_dealer_share": [0.50, 0.60, 0.70],
        }
    )
    model = TenorReopeningMeanBaseline().fit(train)
    test_df = pd.DataFrame({"tenor": ["10-Year"], "is_reopening": [True]})
    preds = model.predict(test_df)
    assert preds.iloc[0] == pytest.approx(model.tenor_means_["10-Year"])


def test_recent_history_baseline_uses_exact_lookback_window():
    train = _train_df()
    model = RecentHistoryBaseline(n_lookback=2).fit(train)
    # 2-Year's most recent 2 (by auction_date): 0.30, 0.40
    assert model.tenor_recent_means_["2-Year"] == pytest.approx(0.35)
    assert model.tenor_n_used_["2-Year"] == 2


def test_recent_history_baseline_uses_all_available_when_fewer_than_lookback():
    train = _train_df()
    model = RecentHistoryBaseline(n_lookback=100).fit(train)
    assert model.tenor_n_used_["2-Year"] == 4
    assert model.tenor_recent_means_["2-Year"] == pytest.approx(0.25)


def test_recent_history_baseline_falls_back_to_global_mean_for_unseen_tenor():
    train = _train_df()
    model = RecentHistoryBaseline(n_lookback=8).fit(train)
    test_df = pd.DataFrame({"tenor": ["30-Year"]})
    preds = model.predict(test_df)
    assert preds.iloc[0] == pytest.approx(model.global_mean_)


def test_recent_history_baseline_is_frozen_not_recomputed_from_test_data():
    """A model fit once must never change its forecast based on what
    test_df contains -- proves the baseline is a genuinely frozen,
    annual constant, not something recomputed per prediction call."""
    train = _train_df()
    model = RecentHistoryBaseline(n_lookback=2).fit(train)
    small_test = pd.DataFrame({"tenor": ["2-Year"]})
    large_test = pd.DataFrame({"tenor": ["2-Year"] * 50})
    preds_small = model.predict(small_test)
    preds_large = model.predict(large_test)
    assert preds_small.iloc[0] == pytest.approx(preds_large.iloc[0])


def test_recent_history_baseline_is_order_invariant_even_with_a_same_day_tie():
    """Acceptance-review regression test: two rows tied at the MOST
    RECENT auction_date, with n_lookback=1 so only ONE of them can be
    selected -- isolates whether tie-breaking depends on input row
    order. Before the fix, swapping which physical row appeared first
    in the input flipped the forecast (0.9 vs 0.3); the deterministic
    `tie_break_col` secondary sort must make both orderings agree.
    """
    base = pd.DataFrame(
        {
            "tenor": ["10-Year"] * 4,
            "is_reopening": [False] * 4,
            "auction_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01"]),
        }
    )
    natural = base.assign(auction_key=["K0", "K1", "K2", "K3"], primary_dealer_share=[0.10, 0.20, 0.30, 0.90])
    # Same rows, same tied date -- but the two tied rows' identities/values are swapped.
    swapped = base.assign(auction_key=["K0", "K1", "K3", "K2"], primary_dealer_share=[0.10, 0.20, 0.90, 0.30])

    m_natural = RecentHistoryBaseline(n_lookback=1).fit(natural)
    m_swapped = RecentHistoryBaseline(n_lookback=1).fit(swapped)

    # Both must deterministically pick the SAME underlying auction (K3, value 0.90)
    # regardless of which physical row arrived first in the input frame.
    assert m_natural.tenor_recent_means_["10-Year"] == pytest.approx(0.90)
    assert m_swapped.tenor_recent_means_["10-Year"] == pytest.approx(0.90)
    assert m_natural.tenor_recent_means_ == m_swapped.tenor_recent_means_


def test_recent_history_baseline_order_invariant_across_many_shuffles_with_ties():
    rows = []
    for i in range(6):
        rows.append(
            {
                "tenor": "10-Year",
                "is_reopening": False,
                "auction_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * i),
                "auction_key": f"A{i}",
                "primary_dealer_share": 0.1 * (i + 1),
            }
        )
    # Two more rows tied with the LAST date above, sitting at the lookback boundary.
    tie_date = rows[-1]["auction_date"]
    rows.append({"tenor": "10-Year", "is_reopening": False, "auction_date": tie_date, "auction_key": "TIE_A", "primary_dealer_share": 0.77})
    rows.append({"tenor": "10-Year", "is_reopening": False, "auction_date": tie_date, "auction_key": "TIE_B", "primary_dealer_share": 0.99})
    df = pd.DataFrame(rows)

    results = set()
    for seed in range(15):
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        model = RecentHistoryBaseline(n_lookback=3).fit(shuffled)
        results.add(round(model.tenor_recent_means_["10-Year"], 12))
    assert len(results) == 1, f"non-deterministic across shuffles: {results}"


def test_recent_history_baseline_raises_without_tie_break_column():
    df = pd.DataFrame(
        {
            "tenor": ["10-Year"],
            "is_reopening": [False],
            "auction_date": pd.to_datetime(["2020-01-01"]),
            "primary_dealer_share": [0.3],
        }
    )
    with pytest.raises(KeyError):
        RecentHistoryBaseline().fit(df)


def test_predict_before_fit_raises():
    for cls in (GlobalMeanBaseline, TenorMeanBaseline, TenorReopeningMeanBaseline, RecentHistoryBaseline):
        with pytest.raises(RuntimeError):
            cls().predict(pd.DataFrame({"tenor": ["2-Year"], "is_reopening": [False]}))
