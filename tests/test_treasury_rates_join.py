from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.treasury_rates_join import (
    NO_COVERAGE_REASON,
    as_of_join,
)


def _rates_wide(n_days: int = 10, start: str = "2026-01-05") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)  # business days only, like the real source
    return pd.DataFrame(
        {
            "rate_date": dates,
            "publication_date": dates,
            "publication_safe_available_date": dates + pd.Timedelta(days=1),
            "2 Yr": [4.30 + 0.01 * i for i in range(n_days)],
        }
    )


def _auctions(rows):
    defaults = {"cusip": "X", "tenor": "2-Year"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_same_day_rate_not_used_for_a_same_day_cutoff():
    """The core Phase 4A point-in-time rule: a rate carrying the same
    calendar date as the cutoff must not be used, because Treasury
    typically publishes by ~6pm ET and the cutoff's own "close of
    business" instant could easily be earlier the same day.
    """
    rates = _rates_wide()
    same_day_cutoff = rates.loc[3, "rate_date"]
    auctions = _auctions([{"cutoff": same_day_cutoff}])
    result = as_of_join(auctions, rates, cutoff_col="cutoff")
    assert result.loc[0, "rate_date"] < same_day_cutoff  # never the same-day rate
    assert result.loc[0, "rate_date"] == rates.loc[2, "rate_date"]  # falls back one day


def test_cutoff_on_safe_available_date_selects_that_days_rate():
    rates = _rates_wide()
    safe_date = rates.loc[3, "publication_safe_available_date"]
    auctions = _auctions([{"cutoff": safe_date}])
    result = as_of_join(auctions, rates, cutoff_col="cutoff")
    assert result.loc[0, "rate_date"] == rates.loc[3, "rate_date"]


def test_never_selects_a_later_rate():
    rates = _rates_wide()
    for cutoff in pd.date_range("2026-01-01", "2026-01-20", freq="2D"):
        auctions = _auctions([{"cutoff": cutoff}])
        result = as_of_join(auctions, rates, cutoff_col="cutoff")
        if result.loc[0, "rates_join_matched"]:
            assert result.loc[0, "publication_safe_available_date"] <= cutoff


def test_no_auction_dropped_for_lacking_coverage():
    rates = _rates_wide(start="2026-06-01")
    auctions = _auctions([{"cutoff": pd.Timestamp("2020-01-01")}])
    result = as_of_join(auctions, rates, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "rates_join_matched"]
    assert result.loc[0, "2 Yr_missing_reason"] == NO_COVERAGE_REASON


def test_rate_age_calendar_and_business_days():
    rates = _rates_wide()
    cutoff = rates.loc[5, "publication_safe_available_date"]
    auctions = _auctions([{"cutoff": cutoff}])
    result = as_of_join(auctions, rates, cutoff_col="cutoff")
    matched_rate_date = result.loc[0, "rate_date"]
    assert result.loc[0, "rate_observation_age_calendar_days"] == (cutoff - matched_rate_date).days
    assert result.loc[0, "rate_observation_age_business_days"] >= 1


def test_announcement_and_pre_auction_cutoffs_can_differ():
    rates = _rates_wide(n_days=15, start="2026-01-05")
    auctions = pd.DataFrame(
        {
            "cusip": ["A"],
            "tenor": ["2-Year"],
            "announcemt_date": [pd.Timestamp("2026-01-07")],
            "auction_date": [pd.Timestamp("2026-01-16")],
        }
    )
    auctions = add_cutoff_dates(auctions)
    ann = as_of_join(auctions, rates, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    pre = as_of_join(auctions, rates, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    assert ann.loc[0, "rate_date"] < pre.loc[0, "rate_date"]


def test_nat_cutoff_does_not_crash_and_is_unmatched():
    rates = _rates_wide()
    auctions = _auctions([{"cutoff": pd.NaT}])
    result = as_of_join(auctions, rates, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "rates_join_matched"]


def test_reserved_column_name_raises():
    rates = _rates_wide()
    auctions = _auctions([{"cutoff": pd.Timestamp("2026-01-10")}])
    auctions["__as_of_join_row_order__"] = 0
    with pytest.raises(ValueError):
        as_of_join(auctions, rates, cutoff_col="cutoff")
