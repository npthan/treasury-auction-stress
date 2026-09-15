"""Centralized time/timezone conventions for this project.

Two distinct notions of "now" are used deliberately, and must not be
conflated (this module exists because Phase 2's acceptance review
found them conflated -- see the Phase 2 acceptance review):

- **Retrieval bookkeeping** -- when did this project's own pipeline
  actually run -- is an operational/audit fact about this software,
  not a market fact. It is recorded in **UTC**, with explicit
  timezone info, via `utc_now_iso()` and `utc_today_date()`. UTC is
  the unambiguous, DST-free choice for it, and it is exactly what is
  stored in `RawArtifactMetadata.retrieval_timestamp_utc` and used to
  name cache files (`retrieval_date`).
- **Treasury market dates** -- e.g. "what is the latest auction date
  that should be considered in scope as of right now" -- are derived
  in `MARKET_TIMEZONE` (`America/New_York`) via `market_today_date()`,
  because the U.S. Treasury auction market itself operates on Eastern
  time. This is directly verified, not assumed: official Treasury
  auction announcements state competitive/noncompetitive closing
  times in Eastern time (e.g. "11:00 AM ET", "1:00 PM ET" -- see the
  primary-source citations in the Phase 2 acceptance review).

A UTC calendar date and an America/New_York calendar date disagree for
several hours around every UTC midnight (which falls in the evening,
U.S. Eastern time) -- using UTC for a "through today" *market* date
default would silently shift the requested `auction_date` upper bound
by a calendar day depending on what hour (UTC) the pipeline happens to
run. This module makes that distinction explicit once, instead of
re-deciding it ad hoc at every call site.

**Neither convention can reclassify a pending auction as completed.**
Completion status is decided solely by whether the API populated an
auction's result fields (see `results_available` in
`treasury_auction_stress.data.normalize`), never by comparing dates --
so a UTC/Eastern boundary disagreement changes which auctions are
*requested*, never whether a requested auction is treated as settled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

MARKET_TIMEZONE = ZoneInfo("America/New_York")

# Verified 2026-09-10 against https://www.newyorkfed.org/aboutthefed/holiday_schedule:
# the New York Fed observes the eleven standard U.S. federal holidays,
# with the same weekend-observance shift pandas' USFederalHolidayCalendar
# already applies (Saturday holiday -> preceding Friday, Sunday holiday
# -> following Monday) -- both independently confirmed for 2026 (e.g.
# Independence Day, July 4 2026, a Saturday, observed Friday July 3).
# Used by treasury_auction_stress.data.dealer_stats_normalize to
# compute a conservative NY Fed publication date (Phase 3); added here
# because time_utils.py is this project's one centralized place for
# calendar conventions, per its own module docstring.
_FED_HOLIDAY_CALENDAR = USFederalHolidayCalendar()


def us_federal_holidays(start: str, end: str) -> pd.DatetimeIndex:
    """U.S. federal holidays (observed dates) in [start, end], per the
    Federal Reserve's own holiday schedule -- see module note above.
    """
    return _FED_HOLIDAY_CALENDAR.holidays(start=start, end=end)


def next_business_day(date: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.Timestamp:
    """The next calendar day after `date` that is neither a weekend day
    nor in `holidays`. `date` itself is never returned, even if it is
    already a business day -- this always advances at least one day.
    """
    candidate = date + pd.Timedelta(days=1)
    holiday_set = set(pd.DatetimeIndex(holidays).normalize())
    while candidate.weekday() >= 5 or candidate.normalize() in holiday_set:
        candidate += pd.Timedelta(days=1)
    return candidate


def next_full_business_day_after(date: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.Timestamp:
    """This project's standard "safe availability" rule, named once and
    reused by every source that needs it (Phase 3's NY Fed Primary
    Dealer Statistics join and Phase 4's Treasury rates/CFTC/RTDSM
    joins): data published/available on `date` is never treated as
    usable until the next **full U.S. business day** after it -- never
    merely `date + 1 calendar day`, which silently produces a Saturday
    availability date for any Friday publication (a Phase 4 acceptance
    review finding: `treasury_rates_normalize.py` and
    `cftc_release_calendar.py` both had this bug; `rtdsm_normalize.py`
    did not, because it already called `next_business_day` here rather
    than adding a raw `Timedelta`).

    A plain synonym for `next_business_day` -- kept as a distinctly
    named function so call sites document *why* they are calling it
    (the project-wide safe-availability rule) rather than reusing a
    generic date-math helper by coincidence, and so future source
    modules have one obvious function to call instead of re-deriving
    subtly different Friday/weekend/holiday logic per module.
    """
    return next_business_day(date, holidays)


def previous_business_day(date: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.Timestamp:
    """The closest calendar day before `date` that is neither a weekend
    day nor in `holidays`. Mirror of `next_business_day`, used for the
    pre-auction cutoff ("the previous business day's close").
    """
    candidate = date - pd.Timedelta(days=1)
    holiday_set = set(pd.DatetimeIndex(holidays).normalize())
    while candidate.weekday() >= 5 or candidate.normalize() in holiday_set:
        candidate -= pd.Timedelta(days=1)
    return candidate


def utc_now_iso(now: datetime | None = None) -> str:
    """Full UTC instant, with explicit timezone offset -- for
    machine-readable retrieval timestamps
    (`RawArtifactMetadata.retrieval_timestamp_utc`).

    `now` is accepted for deterministic testing; production callers
    omit it and get the real current instant.
    """
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).isoformat()


def utc_today_date(now: datetime | None = None) -> str:
    """Today's calendar date in UTC, `YYYY-MM-DD` -- for retrieval-date
    bookkeeping (cache/raw-artifact filenames): an operational fact
    about when this project's pipeline ran, not a market date.
    """
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).date().isoformat()


def market_today_date(now: datetime | None = None) -> str:
    """Today's calendar date in America/New_York, `YYYY-MM-DD` -- for
    deciding the upper bound of an `auction_date` query range ("through
    today"), because the Treasury auction market itself operates on
    Eastern time. Deliberately distinct from `utc_today_date()` -- see
    the module docstring for why conflating the two is a real bug, not
    a style preference.
    """
    current = now or datetime.now(UTC)
    return current.astimezone(MARKET_TIMEZONE).date().isoformat()
