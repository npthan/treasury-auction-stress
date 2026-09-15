"""The CFTC TFF release calendar: distinguishing verified actual
release dates from a conservative, disclosed, inferred rule.

## Phase 4 acceptance review correction (issue 10)

The original implementation assigned every observation Tuesday in a
disruption window the **same** bulk "safe available" date (the
window's final catch-up date). That is wrong whenever CFTC's own
announcements give a genuine **per-report** catch-up schedule: a
report from early in a shutdown is normally released *before* the
last backlogged report, and treating it as unavailable until the very
end of the window is needlessly (and, worse, in the other direction,
was in one case **not conservative enough** -- see below) -- both
directions are point-in-time bugs, and both are fixed here by encoding
`CFTC_REPORT_OVERRIDES`, one entry per individual affected observation
date, reconstructed directly from CFTC's own press releases and its
"Historical Special Announcements" page (`cftc.gov`), not invented.

**A genuine bug this correction fixes**: the original window for the
2023 ION incident stopped at observation 2023-02-21, treating
2023-02-28, 2023-03-07, and 2023-03-14 as ordinary weeks. CFTC's own
Historical Special Announcements page shows those three weeks were
**also** delayed, actually publishing 2023-03-14, 2023-03-16, and
2023-03-21 respectively -- 8-11 days *after* what the ordinary rule
would have computed for them. Any auction cutoff that fell in that gap
would have been (incorrectly) joined to a CFTC report that had not
actually been published yet. `CFTC_REPORT_OVERRIDES` now covers all 7
affected observation weeks (2023-01-31 through 2023-03-14), not 4.

## Method, per window

1. **Ordinary weeks** (the vast majority): a documented, standard rule
   -- Tuesday observation, published the following Friday, 3:30 PM
   Eastern (CFTC's own release-schedule page) -- with a conservative
   holiday shift and the next-full-business-day safety buffer
   (`time_utils.next_full_business_day_after`; see the module-level
   fix note for why this is not simply `+1 calendar day`).
2. **Individual per-report overrides** during 3 known disruptions,
   sourced as follows:

   - **2018-2019 lapse in appropriations**: CFTC's press release
     "CFTC's Release Schedule for Delayed Market Data Reports" (2019)
     states the first catch-up report (for the 2018-12-24 observation
     Monday, itself shifted from the New Year's-week Tuesday) would
     publish 2019-02-01, and thereafter "one report on Tuesday and
     another on Friday of each week until the reports are current."
     This project reconstructs every intermediate date by mechanically
     applying that stated cadence -- and the reconstruction is
     independently **verified self-consistent**: it predicts the
     obs-2019-03-05 report publishes 2019-03-08, which is exactly what
     the same press release separately confirms as the first
     "current" (fully caught-up) report. Flagged
     `PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE`.
   - **2023 ION Cleared Derivatives cyber incident**: CFTC's
     "Historical Special Announcements" page
     (cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements)
     states, for each of 7 affected reports, the exact calendar date
     each was actually released (e.g. "released 2023-02-24, originally
     scheduled 2023-02-03"). These are **directly dated, individually
     confirmed** publication events, not a reconstruction -- flagged
     `PRECISION_VERIFIED_EXACT_DATE`.
   - **2025 lapse in appropriations**: two official CFTC press
     releases give **conflicting** per-report schedules -- "CFTC to
     Resume Publishing COT Reports Wednesday" (initial catch-up plan)
     and "CFTC to Accelerate Publication of Backlogged COT Data" (a
     later, faster revision). For the 6 observation dates where both
     releases agree (2025-09-30 through 2025-11-04), the shared date is
     used, flagged `PRECISION_VERIFIED_EXACT_DATE`. For the 7
     observation dates where the two releases disagree (the
     accelerated release publishes earlier), this project cannot
     confirm which schedule was actually followed for each individual
     report, so it uses the **later** (less favorable, more
     conservative) of the two officially-announced dates -- never
     assuming the accelerated date applied without confirmation --
     flagged `PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED`.

A coarse `DISRUPTION_WINDOW_RANGES` date-range table is retained purely as a
**fallback safety net**: if an observation date falls inside a known
disrupted period but is not found in `CFTC_REPORT_OVERRIDES` (which
would indicate an incomplete override table, not a real "ordinary"
week), it gets a conservative window-level bound instead of silently
falling through to the ordinary Tuesday-Friday rule -- flagged
`PRECISION_INFERRED_EXCEPTION_TAIL`. A dedicated test
(`test_cftc_release_calendar.py`) asserts this fallback path is never
actually reached for any real observation date this project has
downloaded, i.e. the override table is complete for the data in hand.

Sources (verified 2026-09-11): CFTC press releases "CFTC's Release
Schedule for Delayed Market Data Reports" (2019), CFTC's own
"Historical Special Announcements" page (2023 entries), "CFTC to
Resume Publishing COT Reports Wednesday" and "CFTC to Accelerate
Publication of Backlogged COT Data" (2025); see
`artifacts/cftc_positioning_data_quality.md` for the full citation list
and the Phase 4 acceptance review for the per-report override
table. Never inferred from file modification timestamps, retrieval
dates, or API row order -- only from these documented announcements.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from treasury_auction_stress.data.time_utils import (
    next_business_day,
    next_full_business_day_after,
    us_federal_holidays,
)

STANDARD_LAG_DAYS = 3  # Tuesday observation -> Friday publication

PRECISION_VERIFIED_STANDARD = "documented_standard_rule_inferred_for_this_week"
PRECISION_VERIFIED_EXACT_DATE = "verified_exact_publication_date_from_cftc_announcement"
PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE = "reconstructed_from_cftc_documented_cadence_endpoints_verified"
PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED = "conflicting_official_schedules_conservative_later_date_used"
PRECISION_INFERRED_EXCEPTION_TAIL = "documented_window_inferred_tail_fallback_incomplete_override"

# Retained only as a fallback safety net -- see module docstring. If an
# observation date falls in one of these ranges but has no entry in
# CFTC_REPORT_OVERRIDES below, that indicates an incomplete override
# table, not a genuine "ordinary" week.
DISRUPTION_WINDOW_RANGES: tuple[tuple[str, str, str], ...] = (
    ("2018-2019 lapse in appropriations", "2018-12-24", "2019-03-05"),
    ("2023 ION Cleared Derivatives cyber incident", "2023-01-31", "2023-03-14"),
    ("2025 lapse in appropriations", "2025-09-30", "2025-12-23"),
)


@dataclass(frozen=True)
class ReportOverride:
    observation_date: str
    actual_publication_date: str
    precision: str
    exception_type: str
    citation: str
    notes: str = ""


# One entry per individually affected observation date -- see module
# docstring for the source and reconstruction method behind each group.
CFTC_REPORT_OVERRIDES: tuple[ReportOverride, ...] = (
    # --- 2018-2019 lapse in appropriations ---------------------------------
    # Endpoints verified directly from CFTC press release "CFTC's Release
    # Schedule for Delayed Market Data Reports" (2019); intermediate dates
    # reconstructed by mechanically applying its stated "one report Tuesday,
    # one report Friday, each week, in chronological order" cadence --
    # self-consistent (predicts the confirmed 2019-03-08 catch-up date
    # exactly).
    ReportOverride("2018-12-24", "2019-02-01", PRECISION_VERIFIED_EXACT_DATE, "2018-2019 shutdown",
                    "CFTC press release 'CFTC's Release Schedule for Delayed Market Data Reports' (2019)",
                    "First catch-up report; observation itself shifted from the New Year's-week Tuesday to Monday."),
    ReportOverride("2018-12-31", "2019-02-05", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-01-08", "2019-02-08", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-01-15", "2019-02-12", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-01-22", "2019-02-15", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-01-29", "2019-02-19", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-02-05", "2019-02-22", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-02-12", "2019-02-26", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-02-19", "2019-03-01", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-02-26", "2019-03-05", PRECISION_RECONSTRUCTED_FROM_VERIFIED_CADENCE, "2018-2019 shutdown",
                    "Reconstructed from the documented Tue/Fri catch-up cadence"),
    ReportOverride("2019-03-05", "2019-03-08", PRECISION_VERIFIED_EXACT_DATE, "2018-2019 shutdown",
                    "CFTC press release 'CFTC's Release Schedule for Delayed Market Data Reports' (2019)",
                    "Independently confirmed as the first fully 'current' report -- matches the reconstructed cadence exactly."),
    # --- 2023 ION Cleared Derivatives cyber incident -----------------------
    # Every date below is individually, exactly dated on CFTC's own
    # "Historical Special Announcements" page
    # (cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements).
    ReportOverride("2023-01-31", "2023-02-24", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page", "Report originally scheduled 2023-02-03."),
    ReportOverride("2023-02-07", "2023-03-03", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page", "Report originally scheduled 2023-02-10."),
    ReportOverride("2023-02-14", "2023-03-08", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page", "Report originally scheduled 2023-02-17."),
    ReportOverride("2023-02-21", "2023-03-10", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page", "Report originally scheduled 2023-02-24."),
    ReportOverride("2023-02-28", "2023-03-14", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page",
                    "Report originally scheduled 2023-03-03. Acceptance-review correction: the original "
                    "implementation treated this week as ordinary and would have used ~2023-03-06, 8 days early."),
    ReportOverride("2023-03-07", "2023-03-16", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page",
                    "Report originally scheduled 2023-03-10. Acceptance-review correction: previously treated as "
                    "ordinary; would have used ~2023-03-13, 3 days early."),
    ReportOverride("2023-03-14", "2023-03-21", PRECISION_VERIFIED_EXACT_DATE, "2023 ION incident",
                    "CFTC Historical Special Announcements page",
                    "Report originally scheduled 2023-03-17, withheld pending data validation, then released "
                    "2023-03-21 -- the last delayed report; normal cadence resumes after this one. "
                    "Acceptance-review correction: previously treated as ordinary."),
    # --- 2025 lapse in appropriations ---------------------------------------
    # 09-30 through 11-04: identical in both official CFTC press releases.
    ReportOverride("2025-09-30", "2025-11-19", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases 'CFTC to Resume Publishing COT Reports Wednesday' and "
                    "'CFTC to Accelerate Publication of Backlogged COT Data' (2025) -- dates agree"),
    ReportOverride("2025-10-07", "2025-11-21", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases (2025) -- dates agree"),
    ReportOverride("2025-10-14", "2025-11-25", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases (2025) -- dates agree"),
    ReportOverride("2025-10-21", "2025-12-02", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases (2025) -- dates agree"),
    ReportOverride("2025-10-28", "2025-12-05", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases (2025) -- dates agree"),
    ReportOverride("2025-11-04", "2025-12-09", PRECISION_VERIFIED_EXACT_DATE, "2025 shutdown",
                    "CFTC press releases (2025) -- dates agree"),
    # 11-10 through 12-23: the two press releases disagree (the "Accelerate"
    # release proposes an earlier date for each). This project has not
    # confirmed which schedule was actually followed report-by-report, so it
    # uses the LATER (original, slower) date -- never assuming the
    # acceleration applied without confirmation.
    ReportOverride("2025-11-10", "2025-12-12", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2025-12-12; 'Accelerate Publication' release: 2025-12-10 -- later date used"),
    ReportOverride("2025-11-18", "2025-12-16", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2025-12-16; 'Accelerate Publication' release: 2025-12-12 -- later date used"),
    ReportOverride("2025-11-25", "2025-12-19", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2025-12-19; 'Accelerate Publication' release: 2025-12-15 -- later date used"),
    ReportOverride("2025-12-02", "2025-12-23", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2025-12-23; 'Accelerate Publication' release: 2025-12-17 -- later date used"),
    ReportOverride("2025-12-09", "2025-12-30", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2025-12-30; 'Accelerate Publication' release: 2025-12-19 -- later date used"),
    ReportOverride("2025-12-16", "2026-01-06", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2026-01-06; 'Accelerate Publication' release: 2025-12-23 -- later date used"),
    ReportOverride("2025-12-23", "2026-01-09", PRECISION_CONFLICTING_SCHEDULES_LATER_DATE_USED, "2025 shutdown",
                    "'Resume Publishing' release: 2026-01-09; 'Accelerate Publication' release: 2025-12-29 -- later date used"),
)

_OVERRIDE_BY_OBSERVATION_DATE: dict[pd.Timestamp, ReportOverride] = {
    pd.Timestamp(o.observation_date): o for o in CFTC_REPORT_OVERRIDES
}


def _in_known_disruption_window(observation_date: pd.Timestamp) -> str | None:
    for name, start, end in DISRUPTION_WINDOW_RANGES:
        if pd.Timestamp(start) <= observation_date <= pd.Timestamp(end):
            return name
    return None


def compute_release_dates(observation_dates: pd.Series) -> pd.DataFrame:
    """For each Tuesday (or holiday-shifted) observation date, return a
    DataFrame with `nominal_publication_date`,
    `publication_safe_available_date`, `availability_precision`, and
    `release_rule_source` -- never computed from file modification
    timestamps, retrieval dates, or API row order.

    Every date, disrupted or not, goes through the same final step: the
    next full U.S. business day after its actual/nominal publication
    date (`time_utils.next_full_business_day_after`), never a same-day
    or weekend availability date.
    """
    obs = pd.to_datetime(observation_dates)
    if obs.empty:
        return pd.DataFrame(
            columns=[
                "nominal_publication_date",
                "publication_safe_available_date",
                "availability_precision",
                "release_rule_source",
            ]
        )
    holidays = us_federal_holidays(
        start=(obs.min() - pd.Timedelta(days=7)).isoformat(),
        end=(obs.max() + pd.Timedelta(days=21)).isoformat(),
    )
    holiday_set = set(pd.DatetimeIndex(holidays).normalize())

    nominal_pub = []
    safe_available = []
    precision = []
    source = []
    for d in obs:
        override = _OVERRIDE_BY_OBSERVATION_DATE.get(pd.Timestamp(d))
        if override is not None:
            actual_pub = pd.Timestamp(override.actual_publication_date)
            nominal_pub.append(actual_pub)
            safe_available.append(next_full_business_day_after(actual_pub, holidays))
            precision.append(override.precision)
            source.append(override.citation)
            continue

        window_name = _in_known_disruption_window(d)
        if window_name is not None:
            # Fallback safety net -- should never trigger for a real,
            # already-downloaded observation date; see module docstring.
            nominal_pub.append(pd.NaT)
            safe_available.append(next_full_business_day_after(d + pd.Timedelta(days=60), holidays))
            precision.append(PRECISION_INFERRED_EXCEPTION_TAIL)
            source.append(f"observation falls within the known '{window_name}' disruption window but has no "
                          "individual override entry -- conservative 60-day fallback bound applied")
            continue

        nominal_friday = d + pd.Timedelta(days=STANDARD_LAG_DAYS)
        if nominal_friday.normalize() in holiday_set:
            nominal_friday = next_business_day(nominal_friday - pd.Timedelta(days=1), holidays)
        nominal_pub.append(nominal_friday)
        # Phase 4 acceptance review, issue 2: this used to be
        # `nominal_friday + 1 calendar day`, which produces a Saturday
        # safe-available date for the (overwhelmingly common) case where
        # `nominal_friday` itself lands on an ordinary, non-holiday
        # Friday. Fixed to the next FULL business day after the nominal
        # publication date -- Monday in the ordinary case, later if a
        # holiday intervenes.
        safe_available.append(next_full_business_day_after(nominal_friday, holidays))
        precision.append(PRECISION_VERIFIED_STANDARD)
        source.append("CFTC release-schedule page (standard Tuesday-observation/Friday-publication rule)")

    return pd.DataFrame(
        {
            "nominal_publication_date": nominal_pub,
            "publication_safe_available_date": safe_available,
            "availability_precision": precision,
            "release_rule_source": source,
        },
        index=obs.index,
    )
