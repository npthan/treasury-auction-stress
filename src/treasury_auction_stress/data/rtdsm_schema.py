"""Verified schema and configuration for the Federal Reserve Bank of
Philadelphia's Real-Time Data Set for Macroeconomists (RTDSM).

**Never obtained via FRED/ALFRED** -- see `docs/data_source_governance.md`.
Everything here was verified live on 2026-09-11 by downloading the
actual xlsx workbooks from `philadelphiafed.org` and by reading the
Philadelphia Fed's own "General Notes" documentation PDFs -- not
guessed.

## Phase 4 acceptance review corrections (issues 3 and 4)

Two corrections from the original implementation, both because a
single evidence source was generalized beyond what it actually
supports:

1. **Every monthly-vintage variable no longer shares one "day 18"
   nominal-publication rule.** That rule came from Federal Reserve
   Board industrial-production documentation ("release dates have
   varied from the 12th to the 18th") and was, on review, incorrectly
   applied to variables released by entirely different agencies on
   entirely different schedules -- most visibly **ROUTPUT (BEA GDP)**,
   which actually releases in the **23rd-29th** range, not the
   12th-18th range; day 18 would have been an unsafe (too-early) bound
   for it. Every monthly-vintage variable now carries its own
   `monthly_vintage_nominal_day` and `monthly_vintage_precision_label`,
   each cited to that variable's own releasing institution -- see the
   per-variable table below and the Phase 4 acceptance review
   for the full evidence audit.
2. **Quarterly CPI was replaced with monthly-vintage PCPI** ("Consumer
   Price Index (monthly vintages)," `pcpiMvMd.xlsx`, verified live).
   RTDSM offers both: quarterly-vintage `CPI` (coverage from 1965:Q4,
   only 4 vintages/year) and monthly-vintage `PCPI` (coverage from
   1998:M11, 12 vintages/year -- the same cadence as ROUTPUT/IPT/
   EMPLOY/HSTARTS). Evaluated on this project's own ex-ante criteria
   (economic relevance unchanged; vintage granularity materially
   better for a project joining against *weekly* Treasury auctions;
   1998 coverage start is fully sufficient for this project's
   2010-present sample; release-timing evidence equally reproducible)
   -- not for correlation with any auction outcome. See
   the Phase 4 acceptance review for the full documented
   decision.

## A restrained, verified 6-variable set

Six variables, chosen to cover distinct macro dimensions relevant to
Treasury-auction stress (growth, labor x2, inflation, production,
housing) without expanding the set post-hoc:

| Mnemonic | Description | Obs. freq | Vintage freq | Coverage start |
|---|---|---|---|---|
| ROUTPUT | Real GNP/GDP | quarterly | monthly | 1965:M11 |
| RUC | Civilian unemployment rate | monthly | quarterly | 1965:Q4 |
| PCPI | Consumer Price Index (monthly vintages) | monthly | monthly | 1998:M11 |
| IPT | Industrial production, total | monthly | monthly | 1962:M11 |
| EMPLOY | Nonfarm payroll employment | monthly | monthly | 1964:M12 |
| HSTARTS | Housing starts | monthly | monthly | 1968:M2 |

Confirmed live by downloading each xlsx (2026-09-11): a single-sheet
workbook, `DATE` in column A (period label `YYYY:MM` or `YYYY:QQ`),
one column per vintage. **Missing values are blank cells**, read by
pandas as `NaN` -- never a sentinel string. This project reads
`observation_period` as-is; it never imputes, forward-fills, or
zero-fills a blank cell.

## Vintage-date column naming (verified against the RTDSM "General
Notes" PDFs, `gen_doc_GDI.pdf` and `doc_ip.pdf`)

Every column except `DATE` follows `{MNEMONIC}{yy}{P}{p}`, where `yy`
is a **two-digit** vintage year, `P` is the literal `M` (month) or `Q`
(quarter), and `p` is the one- or two-digit vintage month/quarter.
Two-digit years are resolved via `VINTAGE_YEAR_CENTURY_PIVOT` below --
this project's earliest real vintage year is 62 (1962) and this
ingestion runs in 2026, so a pivot of 50 correctly resolves every
vintage year this project will encounter through 2050 without
ambiguity; it will need revisiting only if RTDSM data collection is
still running and this project still exists past 2050.

## Vintage-date timing: variable-specific evidence, not a shared rule

- **RUC** (quarterly-vintage): the Philadelphia Fed's own footnote in
  `gen_doc_GDI.pdf` states these vintages (explicitly naming "the
  unemployment rate ... " alongside NIPA/M1/M2/reserves) were
  "collected ... as they were available on the 15th day of the middle
  month of each quarter" -- a **documented, midmonth information set**
  described precisely that way in the primary source; this project
  treats it as an **exact official collection day**, not merely an
  assumed convention, because the source itself states the day
  explicitly (day 15), distinct from an "around midmonth" approximation.
  precision = `PRECISION_VERIFIED_EXACT_DAY`.
- **ROUTPUT, PCPI, IPT, EMPLOY, HSTARTS** (monthly-vintage): each has
  its **own** `monthly_vintage_nominal_day`, derived from that specific
  variable's own releasing institution's documented schedule -- see
  `VARIABLE_REGISTRY` below and the Phase 4 acceptance review
  for the full per-variable evidence table and citations. None of them
  reuse another variable's evidence.
- Every rule adds the same `PUBLICATION_SAFE_AVAILABILITY_BUFFER_DAYS = 1`
  used everywhere else in this project for same-day-ambiguity safety,
  via `time_utils.next_full_business_day_after`.
"""

from __future__ import annotations

VINTAGE_YEAR_CENTURY_PIVOT = 50  # yy <= 50 -> 2000+yy; yy > 50 -> 1900+yy

QUARTERLY_VINTAGE_NOMINAL_DAY = 15
PUBLICATION_SAFE_AVAILABILITY_BUFFER_DAYS = 1

PRECISION_VERIFIED_EXACT_DAY = "documented_exact_collection_day_15th_of_middle_month_of_quarter"
PRECISION_CONSERVATIVE_BOUND_TEMPLATE = "conservative_bound_from_{institution}_release_schedule"

# Q -> the calendar month number RTDSM calls the "middle month" of that quarter.
QUARTER_TO_MIDDLE_MONTH: dict[int, int] = {1: 2, 2: 5, 3: 8, 4: 11}

# Number of observation periods back that constitutes "one year ago,"
# by observation frequency -- used for a single, uniform year-over-year
# candidate feature rather than a variable-specific set of lags.
YOY_LAG_PERIODS_BY_OBSERVATION_FREQUENCY: dict[str, int] = {"monthly": 12, "quarterly": 4}


class RtdsmVariable:
    __slots__ = (
        "coverage_start_vintage",
        "description",
        "exact_historical_dates_available",
        "known_limitations",
        "mnemonic",
        "monthly_vintage_nominal_day",
        "monthly_vintage_precision_label",
        "observation_frequency",
        "release_date_source_citation",
        "releasing_institution",
        "units",
        "vintage_frequency",
        "xlsx_url",
    )

    def __init__(
        self,
        *,
        mnemonic: str,
        description: str,
        xlsx_url: str,
        observation_frequency: str,
        vintage_frequency: str,
        coverage_start_vintage: str,
        units: str,
        releasing_institution: str,
        release_date_source_citation: str,
        exact_historical_dates_available: bool,
        monthly_vintage_nominal_day: int | None = None,
        monthly_vintage_precision_label: str | None = None,
        known_limitations: str = "",
    ) -> None:
        self.mnemonic = mnemonic
        self.description = description
        self.xlsx_url = xlsx_url
        self.observation_frequency = observation_frequency
        self.vintage_frequency = vintage_frequency
        self.coverage_start_vintage = coverage_start_vintage
        self.units = units
        self.releasing_institution = releasing_institution
        self.release_date_source_citation = release_date_source_citation
        self.exact_historical_dates_available = exact_historical_dates_available
        self.monthly_vintage_nominal_day = monthly_vintage_nominal_day
        self.monthly_vintage_precision_label = monthly_vintage_precision_label
        self.known_limitations = known_limitations


_MEDIA_BASE = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx"

VARIABLE_REGISTRY: tuple[RtdsmVariable, ...] = (
    RtdsmVariable(
        mnemonic="ROUTPUT",
        description="Real GNP/GDP",
        xlsx_url=f"{_MEDIA_BASE}/routputMvQd.xlsx",
        observation_frequency="quarterly",
        vintage_frequency="monthly",
        coverage_start_vintage="1965M11",
        units="Billions of chained dollars (base year varies by vintage)",
        releasing_institution="Bureau of Economic Analysis (BEA)",
        release_date_source_citation=(
            "BEA's own news release schedule (bea.gov/news/schedule), verified live 2026-09-11: Q3 2026 "
            "GDP advance/second/third estimates scheduled for Oct 29 / Nov 25 / Dec 23 -- day-of-month range "
            "23-29, not the 12th-18th range used (incorrectly, before this review) for industrial production."
        ),
        exact_historical_dates_available=False,
        monthly_vintage_nominal_day=30,
        monthly_vintage_precision_label=PRECISION_CONSERVATIVE_BOUND_TEMPLATE.format(institution="bea_gdp"),
        known_limitations=(
            "BEA's exact release day varies release-to-release (23rd-29th observed) and this project has not "
            "verified every historical month's exact day; day 30 (clamped to the true last day of shorter "
            "months) is a conservative bound above the entire observed range, not a per-vintage exact date."
        ),
    ),
    RtdsmVariable(
        mnemonic="RUC",
        description="Civilian unemployment rate",
        xlsx_url=f"{_MEDIA_BASE}/rucQvMd.xlsx",
        observation_frequency="monthly",
        vintage_frequency="quarterly",
        coverage_start_vintage="1965Q4",
        units="Percentage points, seasonally adjusted",
        releasing_institution="Bureau of Labor Statistics (BLS), via RTDSM's own quarterly-vintage collection",
        release_date_source_citation=(
            "Philadelphia Fed 'General Notes' (gen_doc_GDI.pdf) footnote: RUC vintages 'collected ... as they "
            "were available on the 15th day of the middle month of each quarter' -- RTDSM's own stated "
            "collection day, not a BLS release-calendar day."
        ),
        exact_historical_dates_available=True,
    ),
    RtdsmVariable(
        mnemonic="PCPI",
        description="Consumer Price Index (monthly vintages) -- replaces quarterly-vintage CPI, see module docstring",
        xlsx_url=f"{_MEDIA_BASE}/pcpiMvMd.xlsx",
        observation_frequency="monthly",
        vintage_frequency="monthly",
        coverage_start_vintage="1998M11",
        units="Index level, seasonally adjusted",
        releasing_institution="Bureau of Labor Statistics (BLS)",
        release_date_source_citation=(
            "BLS's own CPI release-schedule page (bls.gov/schedule/news_release/cpi.htm), verified live "
            "2026-09-11: CPI 'generally' releases 'in the second full week of the month' (no fixed day; "
            "Tuesday-Thursday historically observed) -- a genuinely different agency and schedule from "
            "industrial production's 12th-18th range."
        ),
        exact_historical_dates_available=False,
        monthly_vintage_nominal_day=16,
        monthly_vintage_precision_label=PRECISION_CONSERVATIVE_BOUND_TEMPLATE.format(institution="bls_cpi"),
        known_limitations=(
            "BLS does not publish CPI on a fixed day of month; day 16 is a conservative bound above the "
            "documented 'second full week' range, not a per-vintage exact date."
        ),
    ),
    RtdsmVariable(
        mnemonic="IPT",
        description="Industrial production, total",
        xlsx_url=f"{_MEDIA_BASE}/iptMvMd.xlsx",
        observation_frequency="monthly",
        vintage_frequency="monthly",
        coverage_start_vintage="1962M11",
        units="Index level, seasonally adjusted (base year varies by vintage)",
        releasing_institution="Federal Reserve Board of Governors",
        release_date_source_citation=(
            "RTDSM's own industrial-production documentation (doc_ip.pdf): 'the Federal Reserve Board "
            "releases its industrial production reports around the middle of the month: ... release dates "
            "have varied from the 12th to the 18th.'"
        ),
        exact_historical_dates_available=False,
        monthly_vintage_nominal_day=18,
        monthly_vintage_precision_label=PRECISION_CONSERVATIVE_BOUND_TEMPLATE.format(institution="federal_reserve_board_ip"),
        known_limitations=(
            "Day 18 is the documented upper bound of the Fed's own stated 12th-18th range for this specific "
            "series -- the one variable in this registry this evidence was actually collected for."
        ),
    ),
    RtdsmVariable(
        mnemonic="EMPLOY",
        description="Nonfarm payroll employment",
        xlsx_url=f"{_MEDIA_BASE}/employMvMd.xlsx",
        observation_frequency="monthly",
        vintage_frequency="monthly",
        coverage_start_vintage="1964M12",
        units="Thousands of employees, seasonally adjusted",
        releasing_institution="Bureau of Labor Statistics (BLS)",
        release_date_source_citation=(
            "BLS's own 2026 Employment Situation release schedule (bls.gov/schedule), verified live "
            "2026-09-11: releases fall between the 2nd and 11th of the month following the reference month "
            "(e.g. Dec 2025 data released Jan 9, 2026; Jan 2026 data released Feb 11, 2026) -- a materially "
            "earlier-in-the-month pattern than industrial production's."
        ),
        exact_historical_dates_available=False,
        monthly_vintage_nominal_day=16,
        monthly_vintage_precision_label=PRECISION_CONSERVATIVE_BOUND_TEMPLATE.format(institution="bls_employment_situation"),
        known_limitations=(
            "Day 16 is a conservative bound above the entire observed 2nd-11th release-day range (verified "
            "for 2026 only); not independently verified for every historical month."
        ),
    ),
    RtdsmVariable(
        mnemonic="HSTARTS",
        description="Housing starts",
        xlsx_url=f"{_MEDIA_BASE}/hstartsMvMd.xlsx",
        observation_frequency="monthly",
        vintage_frequency="monthly",
        coverage_start_vintage="1968M2",
        units="Thousands of housing units, seasonally adjusted",
        releasing_institution="U.S. Census Bureau, jointly with HUD",
        release_date_source_citation=(
            "Census Bureau's New Residential Construction release pattern, verified live 2026-09-11: "
            "preliminary data publishes on 'the 12th business day' of the month following the reference "
            "month -- a business-day count, not a calendar-day range."
        ),
        exact_historical_dates_available=False,
        monthly_vintage_nominal_day=20,
        monthly_vintage_precision_label=PRECISION_CONSERVATIVE_BOUND_TEMPLATE.format(institution="census_hud_new_residential_construction"),
        known_limitations=(
            "The 12th-business-day rule converts to calendar day ~16-18 in most months; day 20 adds extra "
            "margin for a month starting immediately after a weekend/holiday cluster, since this project has "
            "not converted the business-day rule to an exact calendar day for every historical month."
        ),
    ),
)
VARIABLE_BY_MNEMONIC: dict[str, RtdsmVariable] = {v.mnemonic: v for v in VARIABLE_REGISTRY}


def resolve_vintage_year(two_digit_year: int) -> int:
    return 2000 + two_digit_year if two_digit_year <= VINTAGE_YEAR_CENTURY_PIVOT else 1900 + two_digit_year
