"""Verified schema and series-selection information for the Federal
Reserve Bank of New York's Primary Dealer Statistics API.

Everything in this module was captured by directly querying the live
API on 2026-09-10/11 (`curl https://markets.newyorkfed.org/api/pd/...`)
and by reading the Federal Reserve Board's own FR 2004 ("Government
Securities Dealers Reports") form-family documentation, not guessed
from the project plan. See `artifacts/primary_dealer_data_quality.md`
for the full investigation, including two concrete verified findings
this module encodes directly (a mislabeled series description and a
2022 maturity-bucket schema break).

## What the underlying survey is

Primary dealers file the **FR 2004** family of weekly reports with the
Federal Reserve, which the New York Fed (as fiscal agent) publishes as
"Primary Dealer Statistics":

- **FR 2004A** ("Weekly Report of Dealer Positions") -- net positions
  (long minus short) **as of Wednesday close**, at **market value**.
- **FR 2004B** ("Weekly Report of Cumulative Dealer Transactions") --
  cumulative transaction volume **for the calendar week ended
  Wednesday**, at **principal value**.
- **FR 2004C** ("Weekly Report of Dealer Financing and Fails") --
  outstanding financing arrangements (repo, reverse repo, securities
  borrowed, securities lent) and settlement fails **as of Wednesday**,
  reported at the actual funds paid/received (principal value), or at
  fair (market) value of pledged securities when only securities (not
  cash) changed hands.

All three schedules are due "the next business day" after Wednesday,
and the New York Fed's own page states the public release cadence:
"Data are updated on Thursdays at approximately 4:15 p.m. with the
previous week's statistics." See `docs/point_in_time_rules.md` for how
this project turns that into a conservative, date-level publication
rule. Sources: `https://www.federalreserve.gov/apps/reportingforms/Report/Index/FR_2004`
(schedule definitions and as-of/due timing, verified 2026-09-10) and
`https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics`
(release cadence, verified 2026-09-10).

## The API

- List all currently active series: `GET
  https://markets.newyorkfed.org/api/pd/list/timeseries.json` -- no
  auth required (verified: a plain unauthenticated request succeeds).
  Every entry at the last verified pull carries `seriesbreak: "SBN2024"`
  (the API's own internal series-break tag for its *current* code
  set), plus a `keyid` and a free-text `description`.
- Fetch one series' full history: `GET
  https://markets.newyorkfed.org/api/pd/get/{keyid}.json` -- returns
  every `{asofdate, keyid, value}` observation NY Fed has under that
  `keyid`, with no date-range parameter (the only way to narrow a pull
  is to fetch a different `keyid` or slice the response client-side).
- **Verified empirically for every series this project selected**:
  every `asofdate` is a Wednesday, exactly 7 calendar days apart, with
  zero gaps, from `2013-04-03` through the most recent release --
  i.e. this project's own retrieval never observed a skipped week (see
  `artifacts/primary_dealer_data_quality.md` for the exact check).
- **The missing-value token is the literal string `"*"`** -- verified
  directly in live responses for the financing series below (e.g.
  `PDSOOS-UTSETTOT` carries `"*"` for 385 of 701 weeks as of this
  session). This is a *third*, source-specific missing-value
  convention in this project (distinct from the Fiscal Data auctions
  API's literal `"null"` string -- see `schema.API_NULL_TOKEN`) and
  must not be confused with it.

## Why the *current-keyid* API history starts 2013-04-03, not 1998 -- and why that is no longer the whole story

**Correction (Phase 3 acceptance review):** the original text here
claimed pre-2013 history "does not exist" under this API and that this
project made no attempt to retrieve it. That claim was investigated
further, following an explicit instruction not to treat the current
endpoint's behavior as proof of the complete historical record, and
was found to be **incomplete, not correct**.

Every `keyid` this project originally selected (verified individually)
begins exactly `2013-04-03` under the **plain** `/get/{keyid}.json`
endpoint, which lines up with the documented April 2013 FR 2004 form
redesign. But NY Fed's own official menu data (`MENU_URL`, verified
live 2026-09-11) documents an earlier schema period -- `SBP2013`,
"JUL 2001 TO MAR 2013" -- whose data is genuinely retrievable through
a *different*, period-scoped endpoint
(`LEGACY_GET_ENDPOINT_TEMPLATE`), discovered by inspecting the GSDS
UI's own JavaScript bundle rather than assumed. This fully covers this
project's `2010-01-01` auction-history start date. **This project now
extends several (not all) series back to 2001-07-04** -- see the
"Phase 3 acceptance review: pre-2013 historical extension" section
below for exactly which concepts were extended, which were
deliberately left as a genuine gap (a concept that did not exist
before April 2013, e.g. securities borrowed/lent), and which were left
alone because the older data cannot be cleanly separated from TIPS
(a real definitional incompatibility, not a missing feature). The
detailed compatibility matrix lives in
the Phase 3 acceptance review.

## Two later revisions the API still delivers under the *same* keyid

- **2022-01-05**: the Federal Reserve's own FEDS Note ("Insights from
  revised Form FR2004 into primary dealer securities financing and MBS
  activity", 2022-08-05) confirms FR 2004C was revised effective this
  date to report repo/reverse-repo activity across five separate
  counterparty/clearing segments (tri-party ex-GCF, GCF, cleared
  bilateral, uncleared bilateral, sponsored) that did not previously
  exist as separate categories. This project's selected repo/
  reverse-repo series (`PDSORA-UTSETTOT`, `PDSIRRA-UTSETTOT`) are the
  **published total** across whatever segmentation was in force each
  week, so the *keyid* is continuous, but the **methodology underlying
  that total changed on this date** -- flagged via
  `COUNTERPARTY_GRANULARITY_BREAK_2022_01_05` below. This project
  found no equivalent official statement that the securities-borrowed/
  -lent series (`PDSIOSB-UTSETTOT`, `PDSOOS-UTSETTOT`) or the fails
  series changed on this date, and does not claim one.
- **2022-01-05, positions**: `PDPOSGSC-G11L21` (11-21y) and
  `PDPOSGSC-G21` (>21y) **only have data starting 2022-01-05** --
  verified directly (701 weeks for every other selected series, but
  exactly 244 for these two, both starting 2022-01-05). No
  currently-listed keyid covers a combined "more than 11 years"
  Treasury coupon position bucket for the pre-2022 period. This is a
  genuine, currently-unfillable coverage gap for the 20-year and
  30-year tenors' most-specific-bucket feature between this project's
  dealer-data start (2013-04-03) and 2022-01-05 -- not a bug, and not
  silently patched. See `MATURITY_BUCKET_REGIME` below.

## One verified, deliberately-excluded series with an unresolved metadata ambiguity

**Correction (Phase 3 acceptance review):** the original text here
asserted NY Fed had mislabeled `PDTRGST-TOT`. No official NY Fed
source was found that confirms this, so that claim was overreach and
has been withdrawn. What is verified, and all that is claimed now:
`PDTRGST-TOT`'s own API `description` text reads "Total - U.S.
TREASURY INFLATION-PROTECTED SECURITIES (TIPS) DEALER TRANSACTIONS
WITH INTER-DEALER BROKERS + Total - U.S. TREASURY INFLATION-PROTECTED
SECURITIES (TIPS) DEALER TRANSACTIONS WITH OTHERS" -- i.e. its own
label claims to be a TIPS-only total, but its **naming pattern**
(`...GST-TOT`, parallel to `PDPOSGST-TOT`, the verified excl-TIPS
position total) and its **value level** (~$873B avg over the last 10
verified observations, vs. ~$23B for `PDTIPSTOT`, the
unambiguously-named TIPS transactions total, and ~$847B for
`PDGSWOEXTTOT`, the correctly-named excl-TIPS transactions total)
together suggest -- **but do not prove** -- that it may actually carry
Treasury transactions including TIPS rather than a TIPS-only series.
No official NY Fed documentation resolving this discrepancy was found.
This is therefore documented as an **unresolved metadata ambiguity**,
not a confirmed mislabeling: the name/description/value evidence
conflicts, and this project does not know which (if either) is wrong.
This project does **not** select `PDTRGST-TOT` under any
interpretation (it is out of scope whether it means "TIPS only" or
"all Treasuries incl. TIPS," since this project's scope is nominal
coupons excl. TIPS specifically). `PDGSWOEXTTOT` is used instead, and
its excl.-TIPS definition is independently verified two ways, not just
by value level: (1) its value level (~$847B avg, distinct from the
~$23B TIPS-only total) and (2) **structurally**, via NY Fed's own
historical menu data (`https://markets.newyorkfed.org/read?productCode=refdata&targetProductCode=40&group=menu`,
period `SBN2013`, Apr 2013-Dec 2014): `PDGSWOEXTTOT`'s constituent
counterparty-split series (`PDGSIDBEXT`, `PDGSWOEXT`) are listed
directly under a menu node explicitly labeled `"U.S. Treasury
(excluding TIPS)"` (`USTREAS-KEY`), with TIPS transactions
(`PDTRTIPS`) listed as a separate sibling category -- an official,
structural confirmation of scope, not an inference from magnitude
alone. This is recorded here as a direct, concrete instance of this
project's "do not assume a field exists -- verify it" and "do not
assume similarly-named series are historically comparable" rules.
"""

from __future__ import annotations

API_BASE_URL = "https://markets.newyorkfed.org/api/pd"
LIST_ENDPOINT = "/list/timeseries.json"
GET_ENDPOINT_TEMPLATE = "/get/{keyid}.json"
# The historical, period-scoped endpoint discovered during the Phase 3
# acceptance review by inspecting the GSDS UI's own JS bundle
# (`Ao.legacyGetSeriesURL`) -- verified live 2026-09-11. `{period}` is
# one of NY Fed's own schema-period keys (see LEGACY_PERIOD_KEY below,
# sourced from the official menu JSON at MENU_URL).
LEGACY_GET_ENDPOINT_TEMPLATE = "/get/{period}/timeseries/{keyid}.json"
# The official menu describing every historical schema period, its
# exact date range, and which series/categories existed in it --
# verified live 2026-09-11. This is the primary source for every
# historical-period claim in this module.
MENU_URL = "https://markets.newyorkfed.org/read?productCode=refdata&targetProductCode=40&group=menu"

# The literal missing-value token this API uses. NOT the same token as
# the Fiscal Data auctions API (schema.API_NULL_TOKEN == "null").
MISSING_VALUE_TOKEN = "*"

UNITS = "millions of dollars"

VALUATION_MARKET = "market_value"  # FR 2004A (positions)
VALUATION_PRINCIPAL = "principal_value"  # FR 2004B (transactions), FR 2004C (financing/fails)

CATEGORY_POSITION = "position"
CATEGORY_TRANSACTION = "transaction"
CATEGORY_REPO = "financing_repo"
CATEGORY_REVERSE_REPO = "financing_reverse_repo"
CATEGORY_SECURITIES_BORROWED = "securities_borrowed"
CATEGORY_SECURITIES_LENT = "securities_lent"
CATEGORY_FAILS_TO_DELIVER = "fails_to_deliver"
CATEGORY_FAILS_TO_RECEIVE = "fails_to_receive"

# The single continuous regime every selected series (except the two
# noted below) falls under in this API's current keyid namespace.
REGIME_CURRENT_API = "fr2004_api_2013-04-03_present"
# The two long-maturity position buckets: a verified, separate,
# shorter regime -- see the module docstring.
REGIME_LONG_MATURITY_BUCKETS_2022 = "fr2004_api_2022-01-05_present"

REGIME_START_DATES: dict[str, str] = {
    REGIME_CURRENT_API: "2013-04-03",
    REGIME_LONG_MATURITY_BUCKETS_2022: "2022-01-05",
}

# Repo/reverse-repo series whose *published total* is continuous but
# whose underlying counterparty-segment methodology is verified to
# have changed on 2022-01-05 (the FR 2004C revision; see module
# docstring). Not claimed for securities-borrowed/-lent or fails.
COUNTERPARTY_GRANULARITY_BREAK_2022_01_05: frozenset[str] = frozenset(
    {"PDSORA-UTSETTOT", "PDSIRRA-UTSETTOT"}
)

# Tenor -> the FR 2004A maturity-bucket keyid whose remaining-maturity
# range brackets that tenor's *original* term at issuance. This
# mirrors the rest of the project's convention of grouping by
# `original_security_term` ("tenor"), not by a reopening's actual
# remaining maturity at auction time -- see
# treasury_auction_stress.features.eligibility and
# docs/target_specification.md for the same convention applied to
# Dealer Absorption Surprise's tenor grouping. This means a
# well-seasoned reopening's dealer inventory bucket is an
# approximation (its true remaining maturity may have drifted toward
# the next-shorter bucket), not a limitation unique to this feature.
TENOR_TO_MATURITY_BUCKET_KEYID: dict[str, str] = {
    "2-Year": "PDPOSGSC-L2",
    "3-Year": "PDPOSGSC-G2L3",
    "5-Year": "PDPOSGSC-G3L6",
    "7-Year": "PDPOSGSC-G6L7",
    "10-Year": "PDPOSGSC-G7L11",
    "20-Year": "PDPOSGSC-G11L21",
    "30-Year": "PDPOSGSC-G21",
}


class SeriesDefinition:
    """One hand-selected, economically-justified series."""

    __slots__ = (
        "category",
        "keyid",
        "maturity_bucket",
        "rationale",
        "regime_id",
        "stable_name",
        "units",
        "valuation_basis",
    )

    def __init__(
        self,
        *,
        keyid: str,
        stable_name: str,
        category: str,
        maturity_bucket: str | None,
        valuation_basis: str,
        regime_id: str,
        rationale: str,
    ) -> None:
        self.keyid = keyid
        self.stable_name = stable_name
        self.category = category
        self.maturity_bucket = maturity_bucket
        self.units = UNITS
        self.valuation_basis = valuation_basis
        self.regime_id = regime_id
        self.rationale = rationale


# The full, hand-picked, economically-justified series set. Every
# series here is a Treasury-(excluding-TIPS) aggregate, matching this
# project's explicit nominal-coupon-only scope (docs/project_plan.md)
# -- TIPS and FRN dealer-statistics series exist in the API but are
# deliberately not selected, for the same reason TIPS/FRN auctions
# themselves are out of scope.
SELECTED_SERIES: tuple[SeriesDefinition, ...] = (
    # --- Positions (FR 2004A, net long-minus-short, market value) ---
    # Direct measure of "how much of this maturity segment dealers
    # already hold" -- the core candidate for "capacity to absorb more
    # supply in the segment a given auction's tenor falls into."
    SeriesDefinition(
        keyid="PDPOSGS-B",
        stable_name="dealer_net_position_bills",
        category=CATEGORY_POSITION,
        maturity_bucket="bills",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Bill inventory is the dealer balance sheet's most liquid, "
            "fastest-turning-over segment; a large bill position can "
            "indicate general balance-sheet capacity/usage even though "
            "no bill tenor is itself in this project's auction universe."
        ),
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-L2",
        stable_name="dealer_net_position_coupons_le_2y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_le_2y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale="Direct inventory measure for the 2-Year note tenor.",
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G2L3",
        stable_name="dealer_net_position_coupons_2y_3y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_2y_3y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale="Direct inventory measure for the 3-Year note tenor.",
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G3L6",
        stable_name="dealer_net_position_coupons_3y_6y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_3y_6y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale="Direct inventory measure for the 5-Year note tenor.",
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G6L7",
        stable_name="dealer_net_position_coupons_6y_7y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_6y_7y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale="Direct inventory measure for the 7-Year note tenor.",
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G7L11",
        stable_name="dealer_net_position_coupons_7y_11y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_7y_11y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale="Direct inventory measure for the 10-Year note tenor.",
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G11L21",
        stable_name="dealer_net_position_coupons_11y_21y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_11y_21y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_LONG_MATURITY_BUCKETS_2022,
        rationale=(
            "Direct inventory measure for the 20-Year bond tenor. "
            "Only available from 2022-01-05 -- see module docstring."
        ),
    ),
    SeriesDefinition(
        keyid="PDPOSGSC-G21",
        stable_name="dealer_net_position_coupons_gt_21y",
        category=CATEGORY_POSITION,
        maturity_bucket="coupons_gt_21y",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_LONG_MATURITY_BUCKETS_2022,
        rationale=(
            "Direct inventory measure for the 30-Year bond tenor. "
            "Only available from 2022-01-05 -- see module docstring."
        ),
    ),
    SeriesDefinition(
        keyid="PDPOSGST-TOT",
        stable_name="dealer_net_position_total_ex_tips",
        category=CATEGORY_POSITION,
        maturity_bucket="total_ex_tips",
        valuation_basis=VALUATION_MARKET,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Whole-curve aggregate net Treasury (excl. TIPS) position -- "
            "a sanity-check total and a general-balance-sheet-usage "
            "signal independent of any single maturity bucket."
        ),
    ),
    # --- Transactions (FR 2004B, cumulative weekly, principal value) ---
    SeriesDefinition(
        keyid="PDGSWOEXTTOT",
        stable_name="dealer_transaction_volume_total_ex_tips",
        category=CATEGORY_TRANSACTION,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Aggregate weekly Treasury (excl. TIPS) turnover across all "
            "maturities and counterparty types (inter-dealer brokers + "
            "others) -- a market-activity/liquidity signal distinct "
            "from a static position level: a dealer sitting on a large "
            "position amid heavy turnover is in a different situation "
            "than the same position amid a frozen market. "
            "NOTE: PDTRGST-TOT was considered and rejected for this "
            "purpose -- see module docstring."
        ),
    ),
    # --- Financing (FR 2004C, principal/fair value) ---
    SeriesDefinition(
        keyid="PDSORA-UTSETTOT",
        stable_name="dealer_repo_treasury_ex_tips_total",
        category=CATEGORY_REPO,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Total repurchase agreements collateralized by Treasuries "
            "(excl. TIPS) -- dealers pledging Treasury inventory to "
            "borrow cash, i.e. how much of their own inventory is "
            "already financed/levered. A dealer already heavily repo'd "
            "out has less unencumbered balance-sheet room for a new "
            "auction's supply."
        ),
    ),
    SeriesDefinition(
        keyid="PDSIRRA-UTSETTOT",
        stable_name="dealer_reverse_repo_treasury_ex_tips_total",
        category=CATEGORY_REVERSE_REPO,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Total reverse repurchase agreements collateralized by "
            "Treasuries (excl. TIPS) -- dealers lending cash against "
            "Treasury collateral taken in, a measure of dealers' own "
            "cash-financing capacity/matched-book activity in the "
            "Treasury market, and a proxy for collateral scarcity when "
            "compared against the repo (financing-out) side."
        ),
    ),
    SeriesDefinition(
        keyid="PDSIOSB-UTSETTOT",
        stable_name="dealer_securities_borrowed_treasury_ex_tips_total",
        category=CATEGORY_SECURITIES_BORROWED,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Treasury securities dealers have borrowed (as opposed to "
            "financed via repo) -- elevated borrowing of specific "
            "issues is a classic signal of collateral specialness/ "
            "short-covering pressure, relevant to whether dealers can "
            "easily cover a short in a newly-auctioned security."
        ),
    ),
    SeriesDefinition(
        keyid="PDSOOS-UTSETTOT",
        stable_name="dealer_securities_lent_treasury_ex_tips_total",
        category=CATEGORY_SECURITIES_LENT,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Treasury securities dealers have lent out from inventory "
            "-- the complement of securities-borrowed; a large lent-out "
            "position can indicate dealers are earning financing income "
            "on inventory they are not using to distribute supply."
        ),
    ),
    # --- Fails (FR 2004C) ---
    SeriesDefinition(
        keyid="PDFTD-USTET",
        stable_name="dealer_fails_to_deliver_treasury_ex_tips",
        category=CATEGORY_FAILS_TO_DELIVER,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale=(
            "Settlement fails to deliver are a direct, widely-used "
            "indicator of Treasury-market settlement stress and "
            "collateral scarcity -- elevated fails ahead of an auction "
            "can indicate an already-strained plumbing that a new, "
            "large issue would add to."
        ),
    ),
    SeriesDefinition(
        keyid="PDFTR-USTET",
        stable_name="dealer_fails_to_receive_treasury_ex_tips",
        category=CATEGORY_FAILS_TO_RECEIVE,
        maturity_bucket=None,
        valuation_basis=VALUATION_PRINCIPAL,
        regime_id=REGIME_CURRENT_API,
        rationale="The complementary settlement-stress indicator to fails-to-deliver.",
    ),
)

SELECTED_KEYIDS: tuple[str, ...] = tuple(s.keyid for s in SELECTED_SERIES)
SERIES_BY_KEYID: dict[str, SeriesDefinition] = {s.keyid: s for s in SELECTED_SERIES}

# Deliberately-excluded series with an unresolved name/description/
# value ambiguity -- kept here (never fetched) purely so a future
# session doesn't "helpfully" re-add it without re-reading the module
# docstring's investigation. See the softened claim above (Phase 3
# acceptance review): this project does not assert NY Fed mislabeled
# it, only that it is not used under any interpretation.
EXCLUDED_AMBIGUOUS_METADATA_KEYIDS: tuple[str, ...] = ("PDTRGST-TOT",)


# =============================================================================
# Phase 3 acceptance review: pre-2013 historical extension
# =============================================================================
#
# The original Phase 3 review stated dealer data "begins 2013-04-03"
# with no earlier history available. That claim was investigated
# further and found to be **incomplete**: 2013-04-03 is where the
# *current* keyid namespace begins, but NY Fed's own official menu
# data (`MENU_URL` above, verified live 2026-09-11) documents an
# earlier schema period, `SBP2013` ("JUL 2001 TO MAR 2013",
# 2001-07-01 to 2013-03-31), whose data is genuinely retrievable via a
# separate, period-scoped endpoint
# (`LEGACY_GET_ENDPOINT_TEMPLATE`) discovered by inspecting the GSDS
# UI's own JavaScript bundle
# (`https://www.newyorkfed.org/medialibrary/Interactives/markets-data/gsds/gsds-pi-ui/browser/main.js`,
# function `Ao.legacyGetSeriesURL`). This fully covers this project's
# 2010-01-01 auction-history start date. See
# the Phase 3 acceptance review for the complete
# concept-by-concept compatibility matrix; this section is the
# machine-readable result of that investigation.
#
# Verified empirically (live fetch, 2026-09-11): every SBP2013 series
# tested returns exactly 613 weekly (Wednesday) observations,
# 2001-07-04 through 2013-03-27, with no gaps -- the same clean,
# gap-free weekly cadence as the modern API.
#
# The decision, concept by concept, is **not** uniform (per the task's
# framing, it is "some combination" of a genuine gap, an API-choice
# limitation, and a deliberate comparability boundary):
#
# 1. **Directly extended** (same canonical column, same definition,
#    just a longer history): T-Bills and the 3-6 year coupon bucket.
#    Both the pre-2013 keyid (verified via the official menu's
#    "Net Positions" tree) and the modern keyid describe the identical
#    concept -- net (long-minus-short) position, market value, same
#    bucket boundary for the 3-6y case, no bucket boundary at all for
#    bills (a single instrument type). Boundary continuity was checked
#    directly: the last SBP2013 obs and first modern obs for both
#    series differ by an amount well within that series' own routine
#    week-to-week volatility (e.g. bills: $26,770mm -> $46,324mm, a
#    swing smaller than several adjacent weeks in the legacy series
#    itself), not a suspicious jump indicating a unit or definition
#    change.
# 2. **Harmonizable only as a coarser bucket** (never split narrower
#    without source data, per the task's explicit rule): the pre-2013
#    coupon buckets are `<=3y`, `>3<=6y`, `>6<=11y`, `>11y` -- coarser
#    than the modern `<=2y`/`>2<=3y`/`>6<=7y`/`>7<=11y` split. A
#    harmonized `<=3y` feature (summing modern `<=2y`+`>2<=3y`) and a
#    harmonized `>6<=11y` feature (summing modern `>6<=7y`+`>7<=11y`)
#    are exact, defensible aggregations -- never a narrower split of
#    an old broad bucket, only a coarser recombination of new narrow
#    ones, which the task explicitly permits.
# 3. **The long end (>11 years) is a three-regime harmonization**: the
#    pre-2013 combined `>11y` bucket, the 2013-04-03 to 2021-12-29
#    combined `>11y` bucket (keyid `PDPOSGSC-G11` -- discontinued from
#    the live "active series" list in the Jan-2022 redesign, but still
#    directly fetchable via the plain, non-legacy endpoint, verified
#    live 2026-09-11: 457 weeks, 2013-04-03 to 2021-12-29, no gap on
#    either boundary), and the modern `11-21y` + `>21y` split (summed).
#    This answers the task's specific question directly: **yes**, a
#    defensible broad-duration inventory feature can cover both
#    20-Year and 30-Year auctions continuously back to 2001-07-04 --
#    but it cannot and does not pretend to separate 20-Year from
#    30-Year exposure before 2022-01-05.
# 4. **Not extended -- genuine source non-existence, not an API
#    limitation**: securities borrowed and securities lent. The
#    official SBP2013 menu's entire "Financing" tree (verified
#    directly, reproduced in the Phase 3 acceptance review)
#    contains only "Securities In" (reverse repo) and "Securities Out"
#    (repo) -- there is no "Other Financing Activity, Securities
#    Borrowed/Lent" category anywhere in it. The SBN2013 menu (Apr
#    2013 on) is the first period where this category appears. This is
#    not a gap this project's chosen endpoint fails to expose -- the
#    concept itself was not separately reported before April 2013.
# 5. **Not extended -- cannot be cleanly separated from TIPS**:
#    aggregate transaction volume, repo, and reverse-repo. The SBP2013
#    menu's "Transactions" tree nests T-Bills/Coupons/TIIS as sibling
#    "By Security" leaves *under the same* `PDSUSG` ("US Government")
#    parent as the "By Customer" (counterparty) totals this project
#    would otherwise use -- unlike the modern menu, which puts
#    "U.S. Treasury (excluding TIPS)" and "TIPS" as separate top-level
#    siblings. The SBP2013 "Financing" tree's `PDFG` ("US Government")
#    node has no sibling TIPS-specific financing branch at all (unlike
#    "Positions" and "Transactions", which do carry an explicit TIIS
#    branch). Both structural facts indicate the pre-2013 "By
#    Customer" transaction total and the pre-2013 government financing
#    total cannot be cleanly separated from TIPS activity -- extending
#    the canonical excl.-TIPS columns with them would silently splice
#    a different-scope series in, which this project will not do.
# 6. **Not extended -- explicitly a different, broader scope**:
#    settlement fails. The SBP2013 menu labels its only relevant fails
#    category `"US Government (including TIIS)"` -- verbatim, in the
#    official menu's own label text, not inferred. This is a stated,
#    not merely suspected, scope difference from the modern
#    `PDFTD-USTET`/`PDFTR-USTET` (excl.-TIPS) series.
#
# None of the "not extended" decisions above are backfilled,
# interpolated, or spliced into the canonical excl.-TIPS columns. The
# structural missingness before 2013-04-03 for those five series
# remains explicit.

LEGACY_PERIOD_KEY = "SBP2013"
LEGACY_PERIOD_LABEL = "JUL 2001 TO MAR 2013"
LEGACY_PERIOD_START = "2001-07-01"  # per the official menu's own "startdate"
LEGACY_PERIOD_END = "2013-03-31"  # per the official menu's own "enddate"
LEGACY_PERIOD_FIRST_OBSERVATION = "2001-07-04"  # first actual Wednesday obs, verified live

REGIME_LEGACY_SBP2013 = "fr2004_legacy_sbp2013_2001-07-04_to_2013-03-27"
# The discontinued combined long-maturity bucket -- superseded
# 2022-01-05 by PDPOSGSC-G11L21 + PDPOSGSC-G21, but still directly
# fetchable (verified live) via the plain endpoint for its own window.
DISCONTINUED_LONG_BUCKET_KEYID = "PDPOSGSC-G11"
REGIME_DISCONTINUED_LONG_BUCKET = "fr2004_api_2013-04-03_to_2021-12-29_discontinued_2022-01-05"

REGIME_START_DATES[REGIME_LEGACY_SBP2013] = LEGACY_PERIOD_FIRST_OBSERVATION
REGIME_START_DATES[REGIME_DISCONTINUED_LONG_BUCKET] = "2013-04-03"

# Earliest-available-date overrides for canonical stable names whose
# true history spans more than one `SeriesDefinition.regime_id` (a
# single `SeriesDefinition` can only carry one regime id, but these
# two columns are directly extended across the 2013-04-03 boundary --
# see decision 1 above).
REGIME_START_OVERRIDES: dict[str, str] = {
    "dealer_net_position_bills": LEGACY_PERIOD_FIRST_OBSERVATION,
    "dealer_net_position_coupons_3y_6y": LEGACY_PERIOD_FIRST_OBSERVATION,
}


class LegacyExtension:
    """Decision 1: a canonical (already-selected) series directly
    extended backward using a pre-2013 keyid of identical definition.
    """

    __slots__ = ("canonical_keyid", "canonical_stable_name", "comparability_note", "legacy_keyid")

    def __init__(
        self, *, canonical_stable_name: str, canonical_keyid: str, legacy_keyid: str, comparability_note: str
    ) -> None:
        self.canonical_stable_name = canonical_stable_name
        self.canonical_keyid = canonical_keyid
        self.legacy_keyid = legacy_keyid
        self.comparability_note = comparability_note


DIRECT_LEGACY_EXTENSIONS: tuple[LegacyExtension, ...] = (
    LegacyExtension(
        canonical_stable_name="dealer_net_position_bills",
        canonical_keyid="PDPOSGS-B",
        legacy_keyid="PDPUSGTBNOP",
        comparability_note=(
            "Both are 'net position, T-Bills, US Government, market value' "
            "under the same FR 2004A schedule; only the keyid changed at "
            "the April 2013 redesign. Boundary-continuity checked directly."
        ),
    ),
    LegacyExtension(
        canonical_stable_name="dealer_net_position_coupons_3y_6y",
        canonical_keyid="PDPOSGSC-G3L6",
        legacy_keyid="PDPUSGCS36NOP",
        comparability_note=(
            "Identical bucket boundary ('>3 <=6 years') verified against "
            "the official menu for both the pre-2013 and post-2013 "
            "periods; only the keyid changed. Boundary-continuity checked "
            "directly."
        ),
    ),
)
LEGACY_EXTENSION_BY_KEYID: dict[str, LegacyExtension] = {e.legacy_keyid: e for e in DIRECT_LEGACY_EXTENSIONS}


class HarmonizedBucketDefinition:
    """Decisions 2 and 3: a new, coarser, explicitly-named feature
    built by combining a pre-2013 legacy keyid with the exact sum of
    the modern fine-grained buckets it decomposes into. Never splits
    an old broad bucket into narrower ones -- always the reverse
    (coarsening the modern buckets to match the old one).
    """

    __slots__ = (
        "discontinued_middle_keyid",
        "label",
        "legacy_keyid",
        "modern_component_stable_names",
        "modern_component_start_date",
        "stable_name",
        "tenors",
    )

    def __init__(
        self,
        *,
        stable_name: str,
        label: str,
        tenors: tuple[str, ...],
        legacy_keyid: str,
        discontinued_middle_keyid: str | None,
        modern_component_stable_names: tuple[str, ...],
        modern_component_start_date: str,
    ) -> None:
        self.stable_name = stable_name
        self.label = label
        self.tenors = tenors
        self.legacy_keyid = legacy_keyid
        self.discontinued_middle_keyid = discontinued_middle_keyid
        self.modern_component_stable_names = modern_component_stable_names
        self.modern_component_start_date = modern_component_start_date


HARMONIZED_BUCKETS: tuple[HarmonizedBucketDefinition, ...] = (
    HarmonizedBucketDefinition(
        stable_name="dealer_net_position_coupons_le_3y_harmonized",
        label="Coupons <=3 years, harmonized across schema periods",
        tenors=("2-Year", "3-Year"),
        legacy_keyid="PDPUSGCS3LNOP",
        discontinued_middle_keyid=None,
        modern_component_stable_names=(
            "dealer_net_position_coupons_le_2y",
            "dealer_net_position_coupons_2y_3y",
        ),
        modern_component_start_date="2013-04-03",
    ),
    HarmonizedBucketDefinition(
        stable_name="dealer_net_position_coupons_6y_11y_harmonized",
        label="Coupons >6 <=11 years, harmonized across schema periods",
        tenors=("7-Year", "10-Year"),
        legacy_keyid="PDPUSGCS611NOP",
        discontinued_middle_keyid=None,
        modern_component_stable_names=(
            "dealer_net_position_coupons_6y_7y",
            "dealer_net_position_coupons_7y_11y",
        ),
        modern_component_start_date="2013-04-03",
    ),
    HarmonizedBucketDefinition(
        stable_name="dealer_net_position_coupons_gt_11y_harmonized",
        label="Coupons >11 years, harmonized across schema periods",
        tenors=("20-Year", "30-Year"),
        legacy_keyid="PDPUSGCSM11NOP",
        discontinued_middle_keyid=DISCONTINUED_LONG_BUCKET_KEYID,
        modern_component_stable_names=(
            "dealer_net_position_coupons_11y_21y",
            "dealer_net_position_coupons_gt_21y",
        ),
        modern_component_start_date="2022-01-05",
    ),
)
REGIME_START_OVERRIDES.update({b.stable_name: LEGACY_PERIOD_FIRST_OBSERVATION for b in HARMONIZED_BUCKETS})

# A harmonized whole-curve total: exact by construction (sum of every
# legacy coupon bucket + bills == the legacy period's total net
# position; no separate legacy "total" keyid exists in the official
# menu, so this project computes it rather than assuming one).
# Comparable to the modern `PDPOSGST-TOT` (already dollar-reconciled
# in the original Phase 3 work) from 2013-04-03 onward.
HARMONIZED_TOTAL_STABLE_NAME = "dealer_net_position_total_ex_tips_harmonized"
HARMONIZED_TOTAL_MODERN_STABLE_NAME = "dealer_net_position_total_ex_tips"
REGIME_START_OVERRIDES[HARMONIZED_TOTAL_STABLE_NAME] = LEGACY_PERIOD_FIRST_OBSERVATION

# Every legacy-period keyid this project fetches: the two that extend
# a canonical column directly, plus the three fetched purely to build
# a harmonized coarse bucket (never exposed as a "final" feature on
# their own).
LEGACY_ONLY_KEYIDS: tuple[str, ...] = tuple(
    sorted(
        {b.legacy_keyid for b in HARMONIZED_BUCKETS} - {e.legacy_keyid for e in DIRECT_LEGACY_EXTENSIONS}
    )
)
ALL_LEGACY_KEYIDS: tuple[str, ...] = tuple(
    sorted({e.legacy_keyid for e in DIRECT_LEGACY_EXTENSIONS} | set(LEGACY_ONLY_KEYIDS))
)
DISCONTINUED_MIDDLE_KEYIDS: tuple[str, ...] = (DISCONTINUED_LONG_BUCKET_KEYID,)

# Stable names for the three legacy-only raw building-block series and
# the one discontinued middle-piece series -- preserved in the tidy
# long table for full auditability (per the task's "preserve the
# original raw normalized series" rule), even though they are never
# joined onto auctions directly (only the harmonized combination is).
LEGACY_COMPONENT_STABLE_NAMES: dict[str, str] = {
    "PDPUSGCS3LNOP": "dealer_net_position_coupons_le_3y_legacy_component",
    "PDPUSGCS611NOP": "dealer_net_position_coupons_6y_11y_legacy_component",
    "PDPUSGCSM11NOP": "dealer_net_position_coupons_gt_11y_legacy_component",
}
DISCONTINUED_COMPONENT_STABLE_NAMES: dict[str, str] = {
    "PDPOSGSC-G11": "dealer_net_position_coupons_gt_11y_discontinued_component",
}

# Every keyid this project's downloader fetches, across all three
# fetch mechanisms (current/plain, discontinued/plain, legacy/period).
ALL_FETCHED_KEYIDS: tuple[str, ...] = SELECTED_KEYIDS + ALL_LEGACY_KEYIDS + DISCONTINUED_MIDDLE_KEYIDS
