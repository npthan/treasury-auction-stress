"""Verified schema and configuration for the U.S. Treasury Daily Par
Yield Curve Rates ("Constant Maturity Treasury," CMT) source.

Everything here was captured by directly querying the live endpoint on
2026-09-11 (`curl https://home.treasury.gov/resource-center/...`) and
by reading Treasury's own methodology page
(`https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology`),
not guessed. This is the official Treasury Fiscal Data source for
par yield curve rates -- **not** obtained via FRED/ALFRED (see
`docs/data_source_governance.md` for why FRED is excluded this phase).

## The source and endpoint

Official landing page:
`https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve`

Verified CSV export, one calendar year per request, no auth required:

    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv

Each row: `Date` (`MM/DD/YYYY`) plus one column per currently-reported
maturity, value a decimal percent (e.g. `"4.40"`), or an **empty
string** when a maturity was not quoted that day (verified: the `30
Yr` column is present-but-empty for dates during the 2002-2006
issuance suspension -- a genuine, source-native missing-value
convention, not a project invention).

## Verified schema evolution (column set changes by year -- do not
assume a fixed column list)

Confirmed by fetching real CSVs for 1990, 2002, 2006, 2009, 2020, 2024,
and 2026:

| Year(s) | Columns present (in order) |
|---|---|
| 1990 | 3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 30 Yr (no 1 Mo, 2 Mo, 4 Mo, 20 Yr) |
| 2002-2006 | adds 1 Mo, 20 Yr; `30 Yr` column present but **empty** 2002-02-19 through 2006-02-08 (auction suspension) |
| 2009-2019 | 1 Mo, 3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr (no 2 Mo, 4 Mo yet) |
| 2020 | adds `2 Mo` |
| 2022-10-19 onward | adds `4 Mo` (new 17-week bill became a benchmark security) |
| 2025-02-18 onward | adds `1.5 Month` (new 6-week bill became a benchmark security) |

This project's ingestion window (2009-01-01 onward, per
`RATES_START_DATE` below) already has all 7 nominal-coupon tenors
(2/3/5/7/10/20/30-Year) present and non-empty throughout -- the 30-Year
suspension (2002-2006) and the pre-2002 missing 20-Year column both
predate this project's window and are documented here for context
only, not handled operationally.

## Verified methodology regimes (Treasury's own page, quoted directly)

- **`MC` (monotone convex), effective 2021-12-06**: Treasury's own
  page states rates from 2021-12-06 onward use "a monotone convex
  method," replacing the prior "quasi-cubic hermite spline" (`HS`)
  method used through 2021-12-03 (the last business day before).
  Treasury's own page explicitly states rates computed under `HS`
  "remain official" -- i.e. this project must **not** treat the
  regime boundary as something to "correct" older data for; both
  regimes' published values are equally authoritative for their own
  dates, per `docs/point_in_time_rules.md`.
- **20-Year point construction, effective 2020-05-20**: prior to the
  20-Year bond's reintroduction, Treasury's page states the curve
  "used additional inputs that were composites of off-the-run bonds in
  the 20-year range" for that point; from 2020-05-20 the 20-Year point
  reflects the actual, newly-reintroduced 20-Year bond. A `20yr`-
  specific sub-regime, independent of the whole-curve `HS`/`MC` split
  above.
- **4-month bill benchmark, effective 2022-10-19**: the `4 Mo` column
  did not exist before this date (verified: absent from the 2020 CSV,
  present in the 2024 CSV).
- **1.5-month bill benchmark, effective 2025-02-18**: the `1.5 Month`
  column did not exist before this date (verified: absent from the
  2024 CSV, present in the live 2026 CSV).

None of the last two affect this project's selected nominal-coupon
maturities directly, but are recorded for completeness and because a
future phase might use short-end maturities as auxiliary context.

## Publication timing (verified, Treasury's own methodology page)

- Inputs are collected "at or near 3:30 PM" Eastern each trading day.
- "yield curve rates are usually available at Treasury's interest rate
  website by 6:00 PM Eastern Time each trading day," with an explicit
  caveat that publication "may be delayed due to system problems or
  other issues."
- No exact historical intraday publication timestamp is available for
  any date -- only the general, current-day statement above. This
  project therefore applies the same conservative, disclosed-as-a-rule
  (not a fact) pattern already established in Phase 3:
  `publication_date` = the rate's own `rate_date` (same trading day,
  per the ~6:00 PM statement), and `publication_safe_available_date`
  = the next full U.S. business day after `rate_date`
  (`time_utils.next_full_business_day_after`), to resolve the standing
  ambiguity of whether a given cutoff instant on that same calendar day
  fell before or after ~6:00 PM ET. This directly matters for the
  pre-auction cutoff ("previous business day's close"): if that close
  is economically understood as an earlier-in-the-day event (e.g. bond
  market close ~3:00-5:00 PM ET) than Treasury's own ~6:00 PM
  publication of that same day's rate, that same-day rate must not be
  treated as available for that cutoff -- exactly the scenario the
  task's instructions warn about.

  **Phase 4 acceptance review correction**: this rule previously added
  a plain `+1 calendar day`, which produced a **Saturday**
  safe-available date for any Friday `rate_date` (Treasury rates are
  only published on trading days, so every `rate_date` is itself a
  weekday, but `rate_date + 1 calendar day` is not). It now advances to
  the next full business day instead -- Monday in the ordinary case,
  later if a federal holiday intervenes -- never a weekend date.
"""

from __future__ import annotations

CSV_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)
OFFICIAL_LANDING_PAGE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "TextView?type=daily_treasury_yield_curve"
)
METHODOLOGY_PAGE = (
    "https://home.treasury.gov/policy-issues/financing-the-government/"
    "interest-rate-statistics/treasury-yield-curve-methodology"
)

UNITS = "percent"

# This project's ingestion window. 2009-01-01 gives >1 full year of
# lookback before this project's earliest auction (2010-01-01) for
# rolling-window candidate features (max window used: 20 trading days).
RATES_START_DATE = "2009-01-01"

DATE_COLUMN = "Date"

# Every maturity-column label this project has verified appearing in a
# live CSV, mapped to its length in YEARS (a float) -- used for
# adjacent-maturity lookups and documentation. Verifying against a
# fixed, hand-checked list (not inferring from column order) means a
# genuinely new/renamed column is caught as schema drift, not silently
# absorbed.
MATURITY_LABEL_TO_YEARS: dict[str, float] = {
    "1 Mo": 1 / 12,
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

# Direct 1:1 map, tenor label -> CMT maturity column label. Unlike the
# NY Fed dealer-statistics maturity buckets (Phase 3), Treasury's own
# published maturities line up exactly with every nominal-coupon
# tenor -- no bucket approximation is needed here.
TENOR_TO_MATURITY_LABEL: dict[str, str] = {
    "2-Year": "2 Yr",
    "3-Year": "3 Yr",
    "5-Year": "5 Yr",
    "7-Year": "7 Yr",
    "10-Year": "10 Yr",
    "20-Year": "20 Yr",
    "30-Year": "30 Yr",
}

# Ordered by maturity (years) -- used to find each tenor's adjacent
# lower/higher maturity column for candidate features.
MATURITY_LABELS_BY_YEARS_ASC: tuple[str, ...] = tuple(
    sorted(MATURITY_LABEL_TO_YEARS, key=lambda label: MATURITY_LABEL_TO_YEARS[label])
)

# --- Verified methodology regimes ---
REGIME_HS_SPLINE = "treasury_cmt_hs_spline_through_2021-12-03"
REGIME_MC_SPLINE = "treasury_cmt_mc_spline_2021-12-06_present"
WHOLE_CURVE_METHODOLOGY_CHANGE_DATE = "2021-12-06"

REGIME_20Y_COMPOSITE = "treasury_cmt_20y_composite_through_2020-05-19"
REGIME_20Y_REAL_BOND = "treasury_cmt_20y_real_bond_2020-05-20_present"
TWENTY_YEAR_METHODOLOGY_CHANGE_DATE = "2020-05-20"

FOUR_MONTH_INTRODUCED_DATE = "2022-10-19"
ONE_POINT_FIVE_MONTH_INTRODUCED_DATE = "2025-02-18"

# Publication rule (see module docstring): safe availability is the
# next FULL U.S. BUSINESS DAY after the rate's own observation date
# (`time_utils.next_full_business_day_after`) -- never a fixed
# calendar-day offset, which could land on a weekend.
