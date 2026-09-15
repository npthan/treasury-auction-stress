"""Phase 2 auction eligibility: the final analysis-sample rules.

This module takes the already-normalized nominal-coupon subset from
`treasury_auction_stress.data.normalize.nominal_coupon_subset` and
applies the additional rules Phase 2 finalizes on top of it. It never
deletes an "inconvenient" row -- every exclusion or flag applied here
is returned alongside the rows it applies to, so a later session can
see exactly what was left out and why.

## Nominal-coupon inclusion rule (inherited from Phase 1, unchanged)

`security_type in {"Note", "Bond"}` AND `inflation_index_security ==
"No"` AND `floating_rate == "No"` -- see
`treasury_auction_stress.data.schema.NOMINAL_COUPON_SECURITY_TYPES`
and `docs/data_dictionary.md`. This excludes Bills, TIPS, and
2-year FRNs, none of which are in scope for this project (see
`docs/project_plan.md`).

## Supported tenors

All seven nominal coupon tenors: 2/3/5/7/10/20/30-year. The 20-year
bond is structurally different from the other six: it was
discontinued in 1986 and only reintroduced in May 2020, so it has a
much shorter history (77 auctions vs. 190-211 for the other tenors as
of this session) and no pre-2020 observations at all. This is not an
error or something to fill in -- it is a real, permanent limitation of
the 20-year series that must be carried into every later phase's
analysis (e.g. any tenor-level baseline or regime feature for the
20-year bond only ever has data from 2020 onward).

## New issues vs. reopenings

Both are retained as separate observations (per `docs/project_plan.md`
-- a reopening adds to an existing, already-trading CUSIP with known
liquidity characteristics, which is economically different from a new
issue). `is_reopening` (from `normalize.py`) is preserved as a column,
never used to drop rows.

## Completed vs. pending auctions

An auction whose result fields are still the API's `"null"` token
(already converted to missing by `normalize.py`; see
`results_available`) is a **pending/future observation**, not a
failed historical one. `select_analysis_sample` splits these into a
separate `pending` frame rather than mixing them into the settled
analytical sample or silently dropping them. Pending-vs-completed
status is decided **solely** by whether the API populated the result
fields -- never by comparing `auction_date` against "today" in any
timezone (see `treasury_auction_stress.data.time_utils` for the
timezone conventions this project uses elsewhere; a UTC/Eastern
calendar-date disagreement cannot make an upcoming auction look
completed here, because no date comparison is involved in this
decision at all).

## Cancelled or unusual auctions

No cancellation field exists anywhere in the verified live schema
(`treasury_auction_stress.data.schema.EXPECTED_COLUMNS`) -- Treasury's
Fiscal Data auctions dataset appears to only ever record auctions that
occurred (or are scheduled to occur), not ones that were announced and
then cancelled. No cancelled auctions were found or excluded, because
there is no field that could represent one; this is stated explicitly
here so a future session does not assume silence means "checked and
none found by inspecting a cancellation flag" when no such flag
exists to inspect.

## Special auction category: restricted, primary-dealer-only reopenings (verified)

Two settled auctions have an offering amount of exactly $25,000,000 --
far below the tens-of-billions typical for their tenors:

- **2019-06-21**: CUSIP `9128286T2`, 9-Year 11-Month 2-3/8% Note
  (reopening of the 10-year note originally issued 2019-05-15).
- **2021-12-02**: CUSIP `912810TC2`, 19-Year 11-Month 2% Bond
  (reopening of the 20-year bond originally issued 2021-11-30).

This review verified their purpose directly against Treasury's own
official announcement and results press releases (not inferred from
the offering amount alone):

- Announcement: https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/A_20190621_1.pdf
- Results: https://www.treasurydirect.gov/instit/annceresult/press/preanre/2019/R_20190621_1.pdf
- Announcement: https://treasurydirect.gov/instit/annceresult/press/preanre/2021/A_20211202_9.pdf
- Results: https://treasurydirect.gov/instit/annceresult/press/preanre/2021/R_20211202_9.pdf

Both announcements carry **identical, explicit bidding restrictions**,
quoted verbatim (2019 version; the 2021 version differs only in "by
telephone" vs. "by email"):

> "NOTE: Bids shall be submitted by telephone to Treasury's fiscal
> agent the Federal Reserve Bank of New York (FRBNY). Only primary
> dealers, as designated by FRBNY, may submit bids. Each primary
> dealer can submit up to 5 competitive bids. Net long position (NLP)
> reporting will not be required for this coupon. Each competitive bid
> must be at a separate yield. **Customer bids will not be accepted.
> Noncompetitive tenders, including FIMA tenders, will not be
> accepted.** SOMA will be accepted."

Both results releases confirm the outcome that follows mechanically
from this rule: Direct Bidder and Indirect Bidder tendered/accepted
are both exactly $0, and 100% of the (tiny) offering was tendered and
accepted by primary dealers. This is a **verified, resolved finding,
not an unresolved guess**: these are neither operational/system tests
(both settled with real yields, real prices, and real press releases
indistinguishable in format from any other auction) nor ordinary
auctions (ordinary auctions are open to all four bidder categories
through Treasury's standard electronic system; these two were
restricted to primary dealers only, submitted by phone/email directly
to FRBNY). Treasury's announcements do not state *why* a particular
reopening was restricted this way -- that operational reasoning is not
public in the documents this project can access -- so this project
does not speculate about the underlying administrative motive, but the
auction *mechanism* itself is fully and verifiably documented.

**Classification rule.** `noncomp_tenders_accepted == "No"` (a plain
API field, always populated, verified to isolate exactly these two
rows across the entire 2010-2026 nominal-coupon sample and no others)
identifies this category programmatically -- `SPECIAL_AUCTION_TYPE_COL`
is set to `RESTRICTED_PRIMARY_DEALER_ONLY` for these rows and left
missing for every ordinary auction. This is preferred over the
`is_unusually_small_offering` dollar threshold as the *authoritative*
signal, because it is Treasury's own structural rule rather than an
arbitrary cutoff; `is_unusually_small_offering` is retained alongside
it as a secondary, corroborating flag.

**Why these are excluded from the main modeling sample.** A restricted
auction's `primary_dealer_share` is not a measurement of dealer
backstop behavior under normal market conditions -- it is a tautology
imposed by Treasury's own bidding rule (nobody else was allowed to
bid), unrelated to the "Dealer Absorption Surprise" concept this
project is trying to measure. Including them would let two auctions
that cannot possibly reflect weak-demand-driven dealer absorption
masquerade as the two most extreme "surprises" in the entire sample
(which is exactly what happened before this review; see
the Phase 2 acceptance review).

**Disposition, per this review's requirements:**

1. Preserved in the complete normalized dataset -- never dropped by
   `treasury_auction_stress.data.normalize` or by
   `select_analysis_sample`'s `settled` frame.
2. Classified via `SPECIAL_AUCTION_TYPE_COL` (`special_auction_type`).
3. Excluded from the main modeling sample by `select_modeling_sample`,
   an explicit, separate function -- never silently.
4. The exclusion reason is included in `describe_eligibility`'s output
   and in `artifacts/auction_data_profile.md`.
5. `tests/test_eligibility.py` proves both rows are present in
   `settled` (by CUSIP) and absent from `select_modeling_sample`'s
   output.
6. `treasury_auction_stress.features.profile_cli` regenerates all
   figures and summary tables from `select_modeling_sample`'s output,
   not from `settled` directly.
7. `artifacts/auction_data_profile.md` reports Dealer Absorption Surprise
   distribution statistics both including and excluding these two rows.

## Records with incomplete result fields

Verified (see `artifacts/auction_data_quality.md`, regenerated by
`treasury_auction_stress.data.cli`): among settled nominal-coupon
auctions, zero rows have partial missingness across
`primary_dealer_accepted`, `direct_bidder_accepted`,
`indirect_bidder_accepted`, `noncomp_accepted`, `comp_accepted`,
`total_accepted`, `total_tendered`, or `bid_to_cover_ratio` -- an
auction's result fields are either all present or (for a pending
auction) all the `"null"` token. No row needs partial-field handling.

## Duplicate or revised records

The candidate primary key `(cusip, auction_date)` has zero duplicates
across the full verified live pull (`artifacts/auction_data_quality.md`).
This project has only ever retrieved one snapshot of the API per
calendar day (in UTC -- see `treasury_auction_stress.data.time_utils`);
detecting a *revision* to a previously-published auction result (as
opposed to a duplicate row) would require comparing two retrievals of
the same auction taken on different days, which is not yet possible
with a single day's data. This is listed as an open question in
`artifacts/auction_data_profile.md` -- the raw-artifact immutability and
retrieval-date-stamped naming already in place
(`treasury_auction_stress.data.raw_store`) is exactly what will make
this detectable once multiple retrieval-dated snapshots accumulate
across future sessions.

## Analysis start date

`ANALYSIS_START_DATE = "2010-01-01"`, matching `docs/project_plan.md`
and Phase 1's actual pull. Not chosen for any Phase-2-specific reason;
carried forward unchanged. The source has data back to 1979 (see
`docs/data_dictionary.md`) but pulling it is out of this project's
stated scope.
"""

from __future__ import annotations

import pandas as pd

ANALYSIS_START_DATE = "2010-01-01"

# Well below every other offering size observed in the 2010-2026
# nominal-coupon sample (typical sizes are in the tens of billions);
# a secondary, corroborating flag -- see SPECIAL_AUCTION_TYPE_COL for
# the primary, structurally-grounded classification.
SMALL_OFFERING_THRESHOLD = 1_000_000_000

SPECIAL_AUCTION_TYPE_COL = "special_auction_type"
RESTRICTED_PRIMARY_DEALER_ONLY = "restricted_primary_dealer_only_reopening"


def classify_special_auctions(df: pd.DataFrame) -> pd.DataFrame:
    """Add `special_auction_type`: `RESTRICTED_PRIMARY_DEALER_ONLY` for
    auctions where Treasury's own terms excluded customer,
    noncompetitive, and FIMA bids (`noncomp_tenders_accepted == "No"`),
    `pd.NA` for every ordinary auction. See the module docstring for
    the primary-source verification behind this rule.
    """
    out = df.copy()
    is_restricted = out["noncomp_tenders_accepted"] == "No"
    out[SPECIAL_AUCTION_TYPE_COL] = pd.array([pd.NA] * len(out), dtype="string")
    out.loc[is_restricted, SPECIAL_AUCTION_TYPE_COL] = RESTRICTED_PRIMARY_DEALER_ONLY
    return out


def select_analysis_sample(
    nominal_df: pd.DataFrame,
    *,
    start_date: str = ANALYSIS_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the nominal-coupon subset into (settled, pending) analysis
    frames, restricted to `auction_date >= start_date`. Both frames
    carry `is_unusually_small_offering` and `special_auction_type`.

    `settled` is the complete, preserved audit trail -- it still
    contains special (restricted) auctions, tagged, not excluded. Call
    `select_modeling_sample` on its output to get the frame actually
    used for target distributions, figures, and Dealer Absorption
    Surprise. Nothing is dropped except rows before `start_date`.
    """
    df = nominal_df.loc[nominal_df["auction_date"] >= pd.Timestamp(start_date)].copy()
    df["is_unusually_small_offering"] = df["offering_amt"] < SMALL_OFFERING_THRESHOLD
    df = classify_special_auctions(df)

    settled = df.loc[df["results_available"]].copy()
    pending = df.loc[~df["results_available"]].copy()
    return settled, pending


def select_modeling_sample(settled_df: pd.DataFrame) -> pd.DataFrame:
    """The main economic modeling sample: `settled_df` with every
    special (restricted-eligibility) auction excluded, by the explicit
    rule documented in this module's docstring. This is the frame
    `treasury_auction_stress.features.profile_cli` uses for target
    distributions, figures, and Dealer Absorption Surprise -- never
    `settled_df` directly, so a special auction can never silently enter
    the main modeling sample.
    """
    return settled_df.loc[settled_df[SPECIAL_AUCTION_TYPE_COL].isna()].copy()


def describe_eligibility(nominal_df: pd.DataFrame, *, start_date: str = ANALYSIS_START_DATE) -> dict:
    """A small, code-derived summary of the eligibility rules actually
    applied, for embedding in `artifacts/auction_data_profile.md` --
    every number here is computed, not copied from this docstring.
    """
    before_start = nominal_df.loc[nominal_df["auction_date"] < pd.Timestamp(start_date)]
    settled, pending = select_analysis_sample(nominal_df, start_date=start_date)
    modeling_sample = select_modeling_sample(settled)
    special = settled.loc[settled[SPECIAL_AUCTION_TYPE_COL].notna()]

    def _dated_records(frame: pd.DataFrame) -> list[dict]:
        return (
            frame[["auction_date", "tenor", "cusip"]]
            .assign(auction_date=lambda d: d["auction_date"].dt.date.astype(str))
            .to_dict("records")
        )

    return {
        "analysis_start_date": start_date,
        "rows_before_start_date_excluded": len(before_start),
        "settled_analysis_sample_rows": len(settled),
        "modeling_sample_rows": len(modeling_sample),
        "special_auction_rows": len(special),
        "special_auction_type": RESTRICTED_PRIMARY_DEALER_ONLY if len(special) else None,
        "special_auction_records": _dated_records(special),
        "special_auction_exclusion_reason": (
            "Treasury's own auction terms restricted bidding to primary dealers only "
            "(customer, noncompetitive, and FIMA tenders were not accepted); the resulting "
            "100% primary-dealer share is a tautology of the auction's eligibility rule, not "
            "a measurement of dealer backstop behavior under normal market conditions. "
            "Verified against official Treasury announcement and results press releases -- "
            "see treasury_auction_stress.features.eligibility module docstring."
        ),
        "pending_rows": len(pending),
        "pending_auction_dates": sorted(d.isoformat() for d in pending["auction_date"].dt.date.dropna()),
        "unusually_small_offering_rows": _dated_records(
            settled.loc[settled["is_unusually_small_offering"]]
        ),
    }
