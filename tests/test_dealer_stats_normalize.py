from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.dealer_stats_normalize import (
    check_schema_drift,
    compute_publication_dates,
    normalize_dealer_stats,
    pivot_wide,
)
from treasury_auction_stress.data.dealer_stats_schema import (
    REGIME_CURRENT_API,
    REGIME_LEGACY_SBP2013,
    REGIME_LONG_MATURITY_BUCKETS_2022,
    SELECTED_KEYIDS,
)
from treasury_auction_stress.data.time_utils import us_federal_holidays


def test_normalize_parses_official_format_fixture_correctly(dealer_stats_raw_payload):
    long_df, anomalies = normalize_dealer_stats(dealer_stats_raw_payload)
    assert anomalies["unselected_keyids"] == []
    assert anomalies["unparseable_values"] == {}
    assert set(long_df["original_series_code"]) == set(SELECTED_KEYIDS)
    # Every selected series' 12 weeks except the two 2022-regime buckets (7).
    counts = long_df.groupby("original_series_code").size()
    assert counts["PDPOSGS-B"] == 12
    assert counts["PDPOSGSC-G11L21"] == 7
    assert counts["PDPOSGSC-G21"] == 7


def test_missing_values_stay_missing_not_zero(dealer_stats_raw_payload):
    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    lent = long_df.loc[long_df["original_series_code"] == "PDSOOS-UTSETTOT"].set_index("observation_date")
    missing_week = lent.loc[pd.Timestamp("2021-12-01")]
    assert missing_week["is_missing"]
    assert pd.isna(missing_week["value"])
    assert missing_week["value_raw"] == "*"
    assert "source reported the literal '*' token" in missing_week["missing_reason"]

    present_week = lent.loc[pd.Timestamp("2021-12-29")]
    assert not present_week["is_missing"]
    assert present_week["value"] == 121126.0
    assert pd.isna(present_week["missing_reason"])


def test_units_and_valuation_basis_preserved(dealer_stats_raw_payload):
    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    assert (long_df["units"] == "millions of dollars").all()
    positions = long_df.loc[long_df["category"] == "position"]
    assert (positions["valuation_basis"] == "market_value").all()
    financing = long_df.loc[long_df["category"] == "financing_repo"]
    assert (financing["valuation_basis"] == "principal_value").all()


def test_historical_schema_regimes_remain_distinguishable(dealer_stats_raw_payload):
    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    long_bucket = long_df.loc[long_df["original_series_code"] == "PDPOSGSC-G11L21"]
    assert (long_bucket["historical_regime_id"] == REGIME_LONG_MATURITY_BUCKETS_2022).all()
    assert long_bucket["observation_date"].min() == pd.Timestamp("2022-01-05")
    bills = long_df.loc[long_df["original_series_code"] == "PDPOSGS-B"]
    assert (bills["historical_regime_id"] != REGIME_LONG_MATURITY_BUCKETS_2022).all()


def test_schema_break_is_not_silently_crossed_no_pre_2022_rows_for_long_bucket(dealer_stats_raw_payload):
    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    long_bucket = long_df.loc[long_df["original_series_code"] == "PDPOSGSC-G21"]
    assert (long_bucket["observation_date"] >= pd.Timestamp("2022-01-05")).all()
    # Nothing forward-filled or backfilled across the break.
    assert len(long_bucket) == 7


def test_schema_drift_detects_a_missing_selected_keyid(dealer_stats_series_list_body):
    trimmed = {
        "pd": {
            "timeseries": [
                e for e in dealer_stats_series_list_body["pd"]["timeseries"] if e["keyid"] != "PDPOSGS-B"
            ]
        }
    }
    drift = check_schema_drift(trimmed)
    assert drift["missing_keyids"] == ["PDPOSGS-B"]


def test_schema_drift_clean_when_all_selected_keyids_present(dealer_stats_series_list_body):
    drift = check_schema_drift(dealer_stats_series_list_body)
    assert drift["missing_keyids"] == []
    assert drift["checked"] is True


def test_schema_drift_not_checked_when_no_list_payload():
    assert check_schema_drift(None) == {"missing_keyids": [], "checked": False}


def test_compute_publication_dates_nominal_thursday_case():
    obs = pd.Series(pd.to_datetime(["2026-02-04"]))  # an ordinary Wednesday
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0] == pd.Timestamp("2026-02-05")  # Thursday
    assert safe.iloc[0] == pd.Timestamp("2026-02-06")


def test_compute_publication_dates_shifts_past_a_thursday_holiday():
    # Thanksgiving 2025 is Thursday 2025-11-27. The nominal Thursday
    # publication shifts to Friday 2025-11-28 -- which used to make the
    # old `+ 1 calendar day` safe-availability rule land on Saturday
    # 2025-11-29 (a real bug, fixed in Phase 5: see
    # test_compute_publication_dates_friday_publication_does_not_land_on_saturday
    # below). The correct safe-available date is Monday 2025-12-01, the
    # next full business day after Friday.
    obs = pd.Series(pd.to_datetime(["2025-11-26"]))
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0] == pd.Timestamp("2025-11-28")  # Friday, not the holiday Thursday
    assert safe.iloc[0] == pd.Timestamp("2025-12-01")  # Monday, never a weekend day


def test_compute_publication_dates_never_selects_a_date_before_observation():
    obs = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-17", "2025-11-26"]))
    pub, safe = compute_publication_dates(obs)
    assert (pub > obs).all()
    assert (safe > pub).all()


# -- Phase 5: regression tests for the previously-disclosed Phase 3 safe-date
# defect (Phase 4 acceptance review, section 16). The old rule added a plain
# `+ 1 calendar day` to `publication_date`, which is safe for an ordinary
# Thursday-publication week (Thursday + 1 = Friday, a business day) but
# silently produces a Saturday `publication_safe_available_date` whenever a
# holiday shifts `publication_date` itself onto a Friday. Fixed to call the
# shared `next_full_business_day_after` rule instead. -------------------------


def test_compute_publication_dates_ordinary_thursday_publication_is_unaffected():
    # An ordinary week with no holiday: Wednesday obs -> Thursday
    # publication -> Friday safe-available, exactly as before the fix
    # (Thursday + 1 calendar day already lands on a business day).
    obs = pd.Series(pd.to_datetime(["2026-02-04"]))
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0] == pd.Timestamp("2026-02-05")  # Thursday
    assert safe.iloc[0] == pd.Timestamp("2026-02-06")  # Friday


def test_compute_publication_dates_friday_publication_does_not_land_on_saturday():
    # Same scenario as the Thanksgiving test above, isolated as its own
    # dedicated regression test for the exact bug the Phase 5 fix
    # addresses: a Friday publication_date must never produce a
    # Saturday safe-available date.
    obs = pd.Series(pd.to_datetime(["2025-11-26"]))
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0].weekday() == 4  # Friday
    assert safe.iloc[0].weekday() < 5  # never a weekend day
    assert safe.iloc[0] == pd.Timestamp("2025-12-01")


def test_compute_publication_dates_never_lands_on_a_weekend_across_a_full_year():
    # Every Wednesday of 2026, run through the full pipeline -- no
    # publication_safe_available_date should ever fall on a weekend.
    obs = pd.Series(pd.date_range("2026-01-01", "2026-12-31", freq="W-WED"))
    _, safe = compute_publication_dates(obs)
    assert (safe.dt.weekday < 5).all()


def test_compute_publication_dates_federal_holiday_week_safe_date_is_a_business_day():
    # MLK Day 2026 is Monday 2026-01-19 -- not a Thursday, so it does
    # not shift publication_date itself, but this is still a holiday
    # week worth a dedicated check that the safe date stays a business
    # day.
    obs = pd.Series(pd.to_datetime(["2026-01-14"]))  # Wednesday before MLK Day
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0] == pd.Timestamp("2026-01-15")  # Thursday
    assert safe.iloc[0] == pd.Timestamp("2026-01-16")  # Friday, a business day


def test_compute_publication_dates_year_end_boundary_skips_new_years_day():
    # New Year's Day 2026 is Thursday 2026-01-01 -- a Thursday holiday
    # at a year boundary, shifting publication_date to Friday
    # 2026-01-02, whose old-rule `+ 1 calendar day` would have produced
    # Saturday 2026-01-03.
    obs = pd.Series(pd.to_datetime(["2025-12-31"]))  # Wednesday before New Year's Day
    pub, safe = compute_publication_dates(obs)
    assert pub.iloc[0] == pd.Timestamp("2026-01-02")  # Friday, not the holiday Thursday
    assert safe.iloc[0] == pd.Timestamp("2026-01-05")  # Monday, never a weekend day


def test_pivot_wide_shares_one_publication_date_across_all_series(dealer_stats_raw_payload):
    long_df, _ = normalize_dealer_stats(dealer_stats_raw_payload)
    wide = pivot_wide(long_df)
    assert wide["observation_date"].is_monotonic_increasing
    assert wide["publication_safe_available_date"].notna().all()
    # 12 distinct weeks in the fixture window.
    assert len(wide) == 12
    # A week before the long-bucket regime start is NaN there, not absent as a row.
    row = wide.loc[wide["observation_date"] == pd.Timestamp("2021-12-01")].iloc[0]
    assert pd.isna(row["dealer_net_position_coupons_11y_21y"])


def test_us_federal_holidays_used_by_normalize_matches_verified_calendar():
    holidays = us_federal_holidays("2022-01-01", "2022-01-31")
    assert pd.Timestamp("2022-01-17") in pd.DatetimeIndex(holidays)  # MLK Day 2022


def test_historical_extension_merges_legacy_and_modern_rows_into_one_canonical_column(
    dealer_stats_historical_extension_payload,
):
    """Phase 3 acceptance review: `dealer_net_position_bills` must
    appear as ONE canonical stable name spanning both the real legacy
    keyid (`PDPUSGTBNOP`, tagged period `SBP2013`) and the real modern
    keyid (`PDPOSGS-B`, untagged) with no gap and no overlap at the
    verified 2013-04-03 boundary.
    """
    long_df, anomalies = normalize_dealer_stats(dealer_stats_historical_extension_payload)
    assert anomalies == {"unselected_keyids": [], "unparseable_values": {}}

    bills = long_df.loc[long_df["stable_series_name"] == "dealer_net_position_bills"].sort_values(
        "observation_date"
    )
    assert len(bills) == 13  # 8 legacy weeks + 5 modern weeks in this fixture's window
    assert set(bills["original_series_code"]) == {"PDPUSGTBNOP", "PDPOSGS-B"}

    legacy_rows = bills.loc[bills["original_series_code"] == "PDPUSGTBNOP"]
    modern_rows = bills.loc[bills["original_series_code"] == "PDPOSGS-B"]
    assert (legacy_rows["historical_regime_id"] == REGIME_LEGACY_SBP2013).all()
    assert (modern_rows["historical_regime_id"] == REGIME_CURRENT_API).all()
    assert legacy_rows["observation_date"].max() == pd.Timestamp("2013-03-27")
    assert modern_rows["observation_date"].min() == pd.Timestamp("2013-04-03")
    # Exactly 7 days apart across the boundary -- no gap, no overlap.
    gap = modern_rows["observation_date"].min() - legacy_rows["observation_date"].max()
    assert gap == pd.Timedelta(days=7)
    assert bills["value"].notna().all()  # real values on both sides, never fabricated


def test_historical_extension_legacy_only_components_get_their_own_stable_names(
    dealer_stats_historical_extension_payload,
):
    long_df, _ = normalize_dealer_stats(dealer_stats_historical_extension_payload)
    for keyid, stable_name in [
        ("PDPUSGCS3LNOP", "dealer_net_position_coupons_le_3y_legacy_component"),
        ("PDPUSGCS611NOP", "dealer_net_position_coupons_6y_11y_legacy_component"),
        ("PDPUSGCSM11NOP", "dealer_net_position_coupons_gt_11y_legacy_component"),
    ]:
        sub = long_df.loc[long_df["original_series_code"] == keyid]
        assert len(sub) == 8
        assert (sub["stable_series_name"] == stable_name).all()
        assert (sub["historical_regime_id"] == REGIME_LEGACY_SBP2013).all()
        assert sub["observation_date"].max() == pd.Timestamp("2013-03-27")


def test_historical_extension_pivot_wide_has_no_duplicate_or_missing_boundary_week(
    dealer_stats_historical_extension_payload,
):
    long_df, _ = normalize_dealer_stats(dealer_stats_historical_extension_payload)
    wide = pivot_wide(long_df)
    assert wide["observation_date"].is_monotonic_increasing
    assert wide["observation_date"].duplicated().sum() == 0
    assert len(wide) == 13
