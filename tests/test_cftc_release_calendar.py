from __future__ import annotations

import pandas as pd

from treasury_auction_stress.data.cftc_release_calendar import (
    CFTC_REPORT_OVERRIDES,
    PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED,
    PRECISION_INFERRED_EXCEPTION_TAIL,
    PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE,
    PRECISION_VERIFIED_EXACT_DATE,
    PRECISION_VERIFIED_STANDARD,
    compute_release_dates,
)


def test_ordinary_friday_never_produces_a_saturday_safe_date():
    obs = pd.Series(pd.to_datetime(["2026-09-15"]))  # an ordinary Tuesday, no adjacent disruption/holiday
    result = compute_release_dates(obs)
    assert result.iloc[0]["nominal_publication_date"] == pd.Timestamp("2026-09-18")  # Friday
    assert result.iloc[0]["publication_safe_available_date"] == pd.Timestamp("2026-09-21")  # Monday, not Saturday
    assert result.iloc[0]["availability_precision"] == PRECISION_VERIFIED_STANDARD


def test_one_report_per_disruption_exception_type_has_test_coverage():
    """Issue 10: at least one report from each exception period."""
    exception_types = {o.exception_type for o in CFTC_REPORT_OVERRIDES}
    assert exception_types == {"2018-2019 shutdown", "2023 ION incident", "2025 shutdown"}


def test_2018_2019_first_catchup_report_is_verified_exact():
    obs = pd.Series(pd.to_datetime(["2018-12-24"]))
    result = compute_release_dates(obs)
    row = result.iloc[0]
    assert row["nominal_publication_date"] == pd.Timestamp("2019-02-01")
    assert row["availability_precision"] == PRECISION_VERIFIED_EXACT_DATE


def test_2018_2019_intermediate_report_is_reconstructed_not_bulk_dated():
    obs = pd.Series(pd.to_datetime(["2019-01-15"]))
    result = compute_release_dates(obs)
    row = result.iloc[0]
    assert row["nominal_publication_date"] == pd.Timestamp("2019-02-12")
    assert row["availability_precision"] == PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE


def test_2018_2019_last_catchup_report_matches_independently_confirmed_date():
    """The reconstructed cadence must predict the independently
    confirmed 2019-03-08 full-catchup date -- self-consistency check.
    """
    obs = pd.Series(pd.to_datetime(["2019-03-05"]))
    result = compute_release_dates(obs)
    assert result.iloc[0]["nominal_publication_date"] == pd.Timestamp("2019-03-08")


def test_2023_ion_incident_previously_missed_late_weeks_are_now_covered():
    """Acceptance-review-found bug: 2023-02-28, 2023-03-07, and
    2023-03-14 were originally treated as ordinary (non-disrupted)
    weeks and would have been joined to a report that had not actually
    published yet. They must now use their real, individually-dated
    publication dates.
    """
    obs = pd.Series(pd.to_datetime(["2023-02-28", "2023-03-07", "2023-03-14"]))
    result = compute_release_dates(obs)
    assert list(result["nominal_publication_date"].dt.date.astype(str)) == ["2023-03-14", "2023-03-16", "2023-03-21"]
    assert (result["availability_precision"] == PRECISION_VERIFIED_EXACT_DATE).all()


def test_2025_conflicting_schedule_dates_use_the_later_date():
    obs = pd.Series(pd.to_datetime(["2025-11-10"]))
    result = compute_release_dates(obs)
    row = result.iloc[0]
    assert row["nominal_publication_date"] == pd.Timestamp("2025-12-12")  # later of 12-10 / 12-12
    assert row["availability_precision"] == PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED


def test_2025_agreed_schedule_dates_are_verified_exact():
    obs = pd.Series(pd.to_datetime(["2025-09-30"]))
    result = compute_release_dates(obs)
    assert result.iloc[0]["availability_precision"] == PRECISION_VERIFIED_EXACT_DATE


def test_no_override_date_ever_produces_a_weekend_safe_available_date():
    obs = pd.Series(pd.to_datetime([o.observation_date for o in CFTC_REPORT_OVERRIDES]))
    result = compute_release_dates(obs)
    assert (result["publication_safe_available_date"].dt.weekday < 5).all()


def test_window_fallback_never_triggers_for_any_documented_override_observation():
    """Every observation date this project has an override entry for
    must resolve via that override, never fall through to the coarse
    window-level fallback (PRECISION_INFERRED_EXCEPTION_TAIL).
    """
    override_dates = pd.Series([pd.Timestamp(o.observation_date) for o in CFTC_REPORT_OVERRIDES])
    result = compute_release_dates(override_dates)
    assert not (result["availability_precision"] == PRECISION_INFERRED_EXCEPTION_TAIL).any()


def test_window_fallback_still_works_if_an_observation_is_genuinely_undocumented():
    """A synthetic observation date inside a known disruption window but
    deliberately absent from CFTC_REPORT_OVERRIDES must still get a
    conservative fallback bound, never silently fall through to the
    ordinary Tuesday-Friday rule.
    """
    undocumented_date = pd.Timestamp("2019-01-01")  # inside the 2018-2019 window, not a real override entry
    assert undocumented_date not in {pd.Timestamp(o.observation_date) for o in CFTC_REPORT_OVERRIDES}
    result = compute_release_dates(pd.Series([undocumented_date]))
    assert result.iloc[0]["availability_precision"] == PRECISION_INFERRED_EXCEPTION_TAIL
    assert result.iloc[0]["publication_safe_available_date"] > undocumented_date + pd.Timedelta(days=30)


def test_ordinary_weeks_immediately_before_and_after_a_window_are_unaffected():
    obs = pd.Series(pd.to_datetime(["2018-12-11", "2018-12-18", "2019-03-12", "2019-03-19"]))
    result = compute_release_dates(obs)
    assert (result["availability_precision"] == PRECISION_VERIFIED_STANDARD).all()
