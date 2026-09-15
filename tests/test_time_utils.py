from datetime import UTC, datetime

import pandas as pd

from treasury_auction_stress.data.time_utils import (
    MARKET_TIMEZONE,
    market_today_date,
    next_business_day,
    next_full_business_day_after,
    previous_business_day,
    us_federal_holidays,
    utc_now_iso,
    utc_today_date,
)


def test_utc_and_market_dates_agree_during_the_day():
    # 18:00 UTC = 14:00 Eastern (EDT, UTC-4 in September) -- same calendar day.
    midday = datetime(2026, 9, 10, 18, 0, tzinfo=UTC)
    assert utc_today_date(midday) == "2026-09-10"
    assert market_today_date(midday) == "2026-09-10"


def test_utc_and_market_dates_disagree_around_utc_midnight():
    """The exact boundary-disagreement scenario the acceptance review
    was worried about: shortly after UTC midnight, it is still the
    *previous* evening in America/New_York. A pipeline run in this
    window must not silently request one extra day of "future" auction
    data because it used the UTC date as if it were the market date.
    """
    just_after_utc_midnight = datetime(2026, 9, 11, 1, 30, tzinfo=UTC)
    assert utc_today_date(just_after_utc_midnight) == "2026-09-11"
    assert market_today_date(just_after_utc_midnight) == "2026-09-10"


def test_market_today_date_uses_america_new_york():
    assert str(MARKET_TIMEZONE) == "America/New_York"
    instant = datetime(2026, 1, 15, 4, 30, tzinfo=UTC)  # 23:30 ET the prior day (EST, UTC-5)
    assert market_today_date(instant) == "2026-01-14"
    assert utc_today_date(instant) == "2026-01-15"


def test_utc_now_iso_preserves_instant_and_reports_utc_offset():
    eastern_afternoon = datetime(2026, 9, 10, 14, 0, tzinfo=MARKET_TIMEZONE)
    iso = utc_now_iso(eastern_afternoon)
    assert iso.endswith("+00:00")
    assert datetime.fromisoformat(iso) == eastern_afternoon


def test_defaults_produce_real_isoformat_strings_without_a_fixed_now():
    # No `now` passed -- exercises the real datetime.now() path used in
    # production, just checking the output is well-formed.
    assert len(utc_today_date()) == 10
    assert len(market_today_date()) == 10
    assert "T" in utc_now_iso()


def test_us_federal_holidays_matches_verified_2026_schedule():
    # Verified 2026-09-10 against
    # https://www.newyorkfed.org/aboutthefed/holiday_schedule (Phase 3).
    holidays = {d.date().isoformat() for d in us_federal_holidays("2026-01-01", "2026-12-31")}
    assert holidays == {
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Washington's Birthday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day, observed (July 4 is a Saturday)
        "2026-09-07",  # Labor Day
        "2026-10-12",  # Columbus Day
        "2026-11-11",  # Veterans Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    }


def test_next_business_day_skips_thanksgiving():
    holidays = us_federal_holidays("2025-01-01", "2025-12-31")
    wednesday_before_thanksgiving = pd.Timestamp("2025-11-26")
    # Thanksgiving 2025 is Thursday 2025-11-27 -- must skip to Friday.
    assert next_business_day(wednesday_before_thanksgiving, holidays) == pd.Timestamp("2025-11-28")


def test_next_business_day_skips_weekend_when_no_holiday_involved():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    friday = pd.Timestamp("2026-09-11")  # an ordinary Friday, no adjacent holiday
    assert next_business_day(friday, holidays) == pd.Timestamp("2026-09-14")  # Monday


def test_next_business_day_always_advances_even_if_input_is_a_business_day():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    tuesday = pd.Timestamp("2026-09-08")
    assert next_business_day(tuesday, holidays) == pd.Timestamp("2026-09-09")


def test_previous_business_day_skips_weekend():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    monday = pd.Timestamp("2026-09-14")
    assert previous_business_day(monday, holidays) == pd.Timestamp("2026-09-11")  # Friday


def test_previous_business_day_skips_a_holiday():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    day_after_labor_day = pd.Timestamp("2026-09-08")  # Labor Day is Mon 2026-09-07
    assert previous_business_day(day_after_labor_day, holidays) == pd.Timestamp("2026-09-04")  # Friday


# -- next_full_business_day_after: this project's shared "safe availability"
# rule, reused by treasury_rates_normalize.py, cftc_release_calendar.py, and
# rtdsm_normalize.py (Phase 4 acceptance review, issue 2) --------------------


def test_friday_publication_becomes_monday_availability():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    friday = pd.Timestamp("2026-09-11")
    result = next_full_business_day_after(friday, holidays)
    assert result == pd.Timestamp("2026-09-14")  # Monday
    assert result.weekday() < 5  # never a weekend day


def test_friday_publication_with_monday_holiday_skips_to_tuesday():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    friday = pd.Timestamp("2026-01-16")
    # Monday 2026-01-19 is MLK Day -- must not stop there.
    result = next_full_business_day_after(friday, holidays)
    assert result == pd.Timestamp("2026-01-20")  # Tuesday
    assert result.normalize() not in set(holidays)


def test_weekend_boundary_never_returns_a_weekend_day():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    for d in pd.date_range("2026-09-01", "2026-09-30"):
        result = next_full_business_day_after(d, holidays)
        assert result.weekday() < 5, f"{d.date()} -> {result.date()} landed on a weekend"


def test_year_end_boundary_skips_new_years_day():
    holidays = us_federal_holidays("2025-12-01", "2026-01-31")
    wednesday_before_new_year = pd.Timestamp("2025-12-31")
    # New Year's Day 2026-01-01 is a Thursday holiday -- must skip to Friday.
    result = next_full_business_day_after(wednesday_before_new_year, holidays)
    assert result == pd.Timestamp("2026-01-02")


def test_historical_exception_saturday_holiday_observed_preceding_friday():
    """Independence Day 2026 falls on a Saturday, observed the preceding
    Friday (2026-07-03) -- the same federal-holiday weekend-observance
    rule this project's NY Fed publication-date logic (Phase 3) already
    depends on. A Thursday publication must skip both the observed
    holiday and the weekend, landing on the following Monday.
    """
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    thursday = pd.Timestamp("2026-07-02")
    result = next_full_business_day_after(thursday, holidays)
    assert result == pd.Timestamp("2026-07-06")  # Monday


def test_never_returns_the_input_date_even_if_already_a_business_day():
    holidays = us_federal_holidays("2026-01-01", "2026-12-31")
    tuesday = pd.Timestamp("2026-09-08")
    assert next_full_business_day_after(tuesday, holidays) != tuesday
