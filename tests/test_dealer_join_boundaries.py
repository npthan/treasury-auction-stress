"""Phase 3 acceptance-review boundary tests for the point-in-time join.

Covers the specific scenarios the review named: a normal Thursday
announcement, a Thursday publication coinciding with the announcement
date, a holiday-shifted week, a Good-Friday week, the 2013-04-03
schema-transition boundary, the 2022-01-05 long-maturity-bucket start,
a pre-2013 auction, one of the two special dealer-only auctions, and
`NaT` on one or both sides of a comparison. Each test is deliberately
narrow and named after the exact scenario it proves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.data.dealer_stats_normalize import (
    compute_publication_dates,
)
from treasury_auction_stress.data.time_utils import (
    previous_business_day,
    us_federal_holidays,
)
from treasury_auction_stress.features.dealer_candidate_features import (
    add_harmonized_positions,
)
from treasury_auction_stress.features.dealer_join import (
    ANNOUNCEMENT_CUTOFF_COL,
    NO_COVERAGE_REASON,
    STABLE_SERIES_NAMES,
    add_cutoff_dates,
    as_of_join,
)


def _synthetic_extended_wide(n_weeks: int = 1150, start: str = "2001-07-04") -> pd.DataFrame:
    """A synthetic wide table shaped like the real, historically-
    extended dealer table: `dealer_net_position_bills` and
    `dealer_net_position_coupons_3y_6y` populated for the *entire*
    window (directly extended), the 2022-only fine buckets populated
    only from 2022-01-05, and the discontinued/legacy component
    columns populated only in their own disjoint windows -- exactly
    the shape `add_harmonized_positions` expects.
    """
    obs = pd.date_range(start, periods=n_weeks, freq="7D")
    legacy_end = pd.Timestamp("2013-03-27")
    middle_end = pd.Timestamp("2021-12-29")
    long_split_start = pd.Timestamp("2022-01-05")

    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "observation_date": obs,
            "dealer_net_position_bills": 40_000 + rng.normal(0, 5000, n_weeks),
            "dealer_net_position_coupons_3y_6y": 15_000 + rng.normal(0, 3000, n_weeks),
            "dealer_net_position_coupons_le_2y": np.where(
                obs >= pd.Timestamp("2013-04-03"), 20_000 + rng.normal(0, 2000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_2y_3y": np.where(
                obs >= pd.Timestamp("2013-04-03"), 10_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_6y_7y": np.where(
                obs >= pd.Timestamp("2013-04-03"), 8_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_7y_11y": np.where(
                obs >= pd.Timestamp("2013-04-03"), 12_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_11y_21y": np.where(
                obs >= long_split_start, 15_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_gt_21y": np.where(
                obs >= long_split_start, 30_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_total_ex_tips": np.where(
                obs >= pd.Timestamp("2013-04-03"), 150_000 + rng.normal(0, 5000, n_weeks), np.nan
            ),
            # Never extended (Phase 3 acceptance review decision 5) --
            # present only from its real regime start, 2013-04-03.
            "dealer_transaction_volume_total_ex_tips": np.where(
                obs >= pd.Timestamp("2013-04-03"), 800_000 + rng.normal(0, 20000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_le_3y_legacy_component": np.where(
                obs <= legacy_end, 25_000 + rng.normal(0, 2000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_6y_11y_legacy_component": np.where(
                obs <= legacy_end, 18_000 + rng.normal(0, 2000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_gt_11y_legacy_component": np.where(
                obs <= legacy_end, 12_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
            "dealer_net_position_coupons_gt_11y_discontinued_component": np.where(
                (obs > legacy_end) & (obs <= middle_end), 40_000 + rng.normal(0, 1000, n_weeks), np.nan
            ),
        }
    )
    base["publication_date"] = base["observation_date"] + pd.Timedelta(days=1)
    base["publication_safe_available_date"] = base["publication_date"] + pd.Timedelta(days=1)
    return base


def _feature_table() -> pd.DataFrame:
    wide = _synthetic_extended_wide()
    return add_harmonized_positions(wide)


def _auction(**overrides) -> pd.DataFrame:
    defaults = {
        "cusip": "TEST1",
        "tenor": "2-Year",
        "offering_amt": 40_000_000_000.0,
        "special_auction_type": pd.array([pd.NA], dtype="string"),
    }
    defaults.update(overrides)
    return pd.DataFrame({k: [v] for k, v in defaults.items()})


def test_normal_thursday_announcement_selects_the_prior_confirmed_week():
    feat = _feature_table()
    auctions = _auction(announcemt_date=pd.Timestamp("2015-06-11"), auction_date=pd.Timestamp("2015-06-17"))
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    assert result.loc[0, "dealer_join_matched"]
    assert result.loc[0, "publication_safe_available_date"] <= result.loc[0, ANNOUNCEMENT_CUTOFF_COL]


def test_thursday_publication_coinciding_with_announcement_date_is_rejected_same_day():
    feat = _feature_table()
    some_week_pub_date = feat["publication_date"].iloc[100]
    auctions = _auction(
        announcemt_date=some_week_pub_date, auction_date=some_week_pub_date + pd.Timedelta(days=7)
    )
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    # The week whose publication_date equals the cutoff must NOT be
    # selected -- only a week published strictly before the cutoff.
    assert result.loc[0, "observation_date"] < feat["observation_date"].iloc[100]


def test_holiday_shifted_week_publication_date_is_not_the_holiday():
    # Thanksgiving 2025 is Thursday 2025-11-27.
    obs = pd.Series(pd.to_datetime(["2025-11-26"]))
    holidays = us_federal_holidays("2025-01-01", "2025-12-31")
    pub, _safe = compute_publication_dates(obs)
    assert pub.iloc[0] not in set(pd.DatetimeIndex(holidays))
    assert pub.iloc[0] == pd.Timestamp("2025-11-28")


def test_good_friday_is_not_treated_as_a_federal_holiday_disclosed_divergence():
    """Documented, disclosed limitation: Good Friday 2026-04-03 is a
    SIFMA-recommended bond-market close but is NOT a Federal Reserve
    holiday, so `previous_business_day` does not skip it. This test
    exists to prove the behavior is exactly what the documentation
    claims, not to endorse it as fully precise.
    """
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    good_friday_2026 = pd.Timestamp("2026-04-03")
    assert good_friday_2026 not in set(pd.DatetimeIndex(holidays))
    # The following Monday's "previous business day" still lands on
    # Good Friday itself, since this project's calendar doesn't know
    # it's a bond-market holiday.
    monday_after = pd.Timestamp("2026-04-06")
    assert previous_business_day(monday_after, holidays) == good_friday_2026


def test_schema_transition_boundary_2013_04_03_extended_series_available_both_sides():
    feat = _feature_table()
    before = _auction(
        cusip="BEFORE", announcemt_date=pd.Timestamp("2013-03-25"), auction_date=pd.Timestamp("2013-03-28")
    )
    after = _auction(
        cusip="AFTER", announcemt_date=pd.Timestamp("2013-04-08"), auction_date=pd.Timestamp("2013-04-11")
    )
    auctions = pd.concat([before, after], ignore_index=True)
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    # Directly-extended series: available on both sides of the boundary.
    assert result["dealer_net_position_bills"].notna().all()
    # Never-extended series: NaN before, present after.
    assert pd.isna(result.loc[0, "dealer_net_position_coupons_le_2y"])
    assert pd.notna(result.loc[1, "dealer_net_position_coupons_le_2y"])
    assert "series_regime_not_yet_started" in result.loc[0, "dealer_net_position_coupons_le_2y_missing_reason"]


def test_2022_long_bucket_start_fine_bucket_missing_but_harmonized_present_before():
    feat = _feature_table()
    before_2022 = _auction(
        cusip="LONG-BEFORE",
        tenor="30-Year",
        announcemt_date=pd.Timestamp("2018-08-06"),
        auction_date=pd.Timestamp("2018-08-09"),
    )
    after_2022 = _auction(
        cusip="LONG-AFTER",
        tenor="30-Year",
        announcemt_date=pd.Timestamp("2022-08-01"),
        auction_date=pd.Timestamp("2022-08-11"),
    )
    auctions = pd.concat([before_2022, after_2022], ignore_index=True)
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    # Fine-grained >21y bucket: missing before 2022-01-05, present after.
    assert pd.isna(result.loc[0, "dealer_net_position_coupons_gt_21y"])
    assert pd.notna(result.loc[1, "dealer_net_position_coupons_gt_21y"])
    # Harmonized >11y bucket: present on BOTH sides (this is the point).
    assert pd.notna(result.loc[0, "dealer_net_position_coupons_gt_11y_harmonized"])
    assert pd.notna(result.loc[1, "dealer_net_position_coupons_gt_11y_harmonized"])


def test_pre_2013_auction_now_matches_and_gets_extended_series_values():
    feat = _feature_table()
    auctions = _auction(
        cusip="PRE2013", announcemt_date=pd.Timestamp("2011-06-13"), auction_date=pd.Timestamp("2011-06-16")
    )
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    assert result.loc[0, "dealer_join_matched"]
    assert pd.notna(result.loc[0, "dealer_net_position_bills"])
    assert pd.isna(result.loc[0, "dealer_net_position_bills_missing_reason"])  # present, so no reason at all
    # A never-extended series correctly still reports its own,
    # series-specific "regime not started" reason, not NO_COVERAGE_REASON.
    assert "series_regime_not_yet_started" in result.loc[0, "dealer_transaction_volume_total_ex_tips_missing_reason"]


def test_one_of_the_two_special_auctions_joins_without_special_casing():
    """The join layer has no concept of "special auction" -- that is
    an eligibility-layer classification (Phase 2). This proves the
    join treats a real special auction (2019-06-21, CUSIP 9128286T2)
    exactly like any other row: it still gets cutoffs, a match
    attempt, and per-series reasons, with no crash or silent drop.
    """
    feat = _feature_table()
    auctions = _auction(
        cusip="9128286T2",
        tenor="10-Year",
        offering_amt=25_000_000.0,
        announcemt_date=pd.Timestamp("2019-06-21"),
        auction_date=pd.Timestamp("2019-06-21"),
        special_auction_type=pd.array(["restricted_primary_dealer_only_reopening"], dtype="string"),
    )
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    assert len(result) == 1
    assert result.loc[0, "special_auction_type"] == "restricted_primary_dealer_only_reopening"
    assert result.loc[0, "dealer_join_matched"]


def test_nat_cutoff_does_not_match_and_does_not_crash():
    feat = _feature_table()
    auctions = _auction(cusip="NATCUTOFF")
    auctions["cutoff"] = pd.NaT
    result = as_of_join(auctions, feat, cutoff_col="cutoff")
    assert len(result) == 1
    assert not result.loc[0, "dealer_join_matched"]
    for name in STABLE_SERIES_NAMES:
        if f"{name}_missing_reason" in result.columns:
            assert result.loc[0, f"{name}_missing_reason"] == NO_COVERAGE_REASON


def test_nat_observation_dates_in_dealer_table_do_not_match_a_real_cutoff():
    feat = _feature_table()
    feat_with_nat_row = pd.concat(
        [
            feat,
            pd.DataFrame(
                {
                    "observation_date": [pd.NaT],
                    "publication_date": [pd.NaT],
                    "publication_safe_available_date": [pd.NaT],
                }
            ),
        ],
        ignore_index=True,
    )
    auctions = _auction(
        cusip="REALCUTOFF", announcemt_date=pd.Timestamp("2015-06-11"), auction_date=pd.Timestamp("2015-06-17")
    )
    auctions = add_cutoff_dates(auctions)
    result = as_of_join(auctions, feat_with_nat_row, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    # A NaT row in the dealer table must never be selected over a real,
    # earlier, valid release.
    assert pd.notna(result.loc[0, "observation_date"])
