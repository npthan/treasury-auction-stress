"""Phase 2 target formulas: bidder-category shares and bid-to-cover.

This module operates on the already-normalized nominal-coupon table
produced by `treasury_auction_stress.data.normalize` (dates parsed,
numbers parsed as nullable ``Float64``, the API's literal ``"null"``
token already converted to a real missing value). It does not read raw
data and does not decide auction eligibility (see
`treasury_auction_stress.features.eligibility` for that) -- it only
turns already-eligible rows into the four candidate outcome fields
described in `docs/target_specification.md`.

## The denominator problem this module exists to solve

The API's `total_accepted` field is **not** simply the sum of the
primary-dealer/direct-bidder/indirect-bidder/noncompetitive amounts.
Verified by exact-dollar reconciliation over the full 2010+ settled
nominal-coupon sample (1281 rows, zero exceptions):

    total_accepted == comp_accepted + noncomp_accepted
                       + fima_noncomp_accepted + soma_accepted

    comp_accepted   == primary_dealer_accepted + direct_bidder_accepted
                        + indirect_bidder_accepted

`soma_accepted` (Federal Reserve System Open Market Account add-ons --
the Fed rolling its own maturing holdings into the new issue) is
awarded **outside** competitive bidding, on top of the publicly
announced offering amount, and represents the Fed's own portfolio
operation rather than any market participant's demand. It is excluded
from every share's denominator.

`fima_noncomp_accepted` (Foreign and International Monetary Authorities
noncompetitive add-ons) is a different kind of quantity, confirmed by
reading the actual Treasury auction-results press releases for two
auctions this project investigated directly (see
the Phase 2 acceptance review for the full text and URLs):
Treasury's own results release reports a "Subtotal" line equal to
`Competitive + Noncompetitive + FIMA (Noncompetitive)`, and its own
footnoted Bid-to-Cover Ratio is computed as that Subtotal's tendered
amount over that Subtotal's accepted amount -- i.e. **Treasury's own
published methodology includes FIMA, and excludes only SOMA**, from
the denominator it uses. Verified over the full settled sample:
recomputing the ratio this way reconciles `bid_to_cover_ratio` to
Treasury's published figure **exactly, for every one of 1281 rows**
(zero exceptions, not just "within rounding") -- see
`reconcile_bid_to_cover` and the Phase 2 acceptance review.
An earlier version of this module excluded FIMA as well as SOMA, which
produced a spurious ~0.01-0.02 "unexplained residual" in the
bid-to-cover reconciliation -- that residual was this formula error,
not rounding noise, and is why FIMA is included below.

**This project therefore uses `public_accepted_amount` (`comp_accepted
+ noncomp_accepted + fima_noncomp_accepted`) as the denominator for all
bidder-category shares and for bid-to-cover -- matching Treasury's own
"Subtotal" exactly.** This amount is verified to partition exactly (to
the dollar, over the full settled sample) into five categories: primary
dealer, direct bidder, indirect bidder, noncompetitive, and FIMA
(noncompetitive). `total_accepted` remains available on the table
unchanged, for anyone who wants the SOMA-inclusive headline figure
Treasury itself also reports.

## Bid-to-cover: the API already provides it, and it now reconciles exactly

`bid_to_cover_ratio` is already a precomputed field. This module
verified it equals, once rounded to 2 decimal places:

    bid_to_cover_ratio == round(
        (comp_tendered + noncomp_accepted + fima_noncomp_accepted)
        / public_accepted_amount,
        2,
    )

for all 1281 settled rows in the sample, with zero exceptions -- see
the Phase 2 acceptance review for the reconciliation table and
the primary-source confirmation (Treasury's own auction-results press
releases show this exact "Subtotal" arithmetic, including a worked
example: "Bid-to-Cover Ratio: $85,850,000/$25,000,400 = 3.43"). This
module keeps the API's own `bid_to_cover_ratio` as the primary field
and exposes the from-scratch calculation (`bid_to_cover_calculated`)
purely as an audit cross-check -- the two are now expected to be
identical (after rounding), so any future discrepancy is a genuine
finding worth investigating, not expected noise.

## Treatment of noncompetitive bids (domestic and FIMA)

Both domestic noncompetitive bids (small, fixed-price orders, mostly
retail, always accepted in full up to a cap) and FIMA noncompetitive
bids (foreign and international monetary authorities, submitted
through the Federal Reserve Bank of New York, also always accepted in
full) are **included in the denominator** (`public_accepted_amount`)
because they represent real, allocated demand for the auctioned
security -- but neither is included in any of the three primary
bidder-category numerators (`primary_dealer_share`,
`direct_bidder_share`, `indirect_bidder_share`); each is its own
diagnostic category (`noncompetitive_share`, `fima_share`).

## Missing vs. zero

Every share is computed with ordinary nullable-float division, so:

- If the underlying auction hasn't settled yet (`total_accepted`,
  `comp_accepted`, etc. are still the API's `"null"` token, already
  converted to `pd.NA` by `normalize.py`), the resulting share is
  `pd.NA` -- not zero, not silently dropped, not imputed. See the
  `results_available` column (from `normalize.py`) to distinguish
  "not yet known" from "known and equal to zero."
- A real zero numerator is preserved as `0.0`. This does happen for
  `direct_bidder_share` and `indirect_bidder_share` (exactly the same 2
  auctions in the verified 2010-2026 history, both restricted to
  primary-dealer-only bidding by Treasury's own auction terms -- see
  `treasury_auction_stress.features.eligibility`) but **never** for
  `primary_dealer_share` in that same history -- consistent with
  primary dealers' contractual obligation to bid at every auction (see
  `docs/target_specification.md`).
"""

from __future__ import annotations

import pandas as pd

# Which raw "accepted" column is the numerator for each candidate
# bidder-category share. All three share `public_accepted_amount` as
# their denominator (see module docstring for why).
SHARE_NUMERATOR_FIELDS: dict[str, str] = {
    "primary_dealer_share": "primary_dealer_accepted",
    "direct_bidder_share": "direct_bidder_accepted",
    "indirect_bidder_share": "indirect_bidder_accepted",
}

# Diagnostic-only shares: not candidate research outcomes, but useful
# for the partition identity (dealer + direct + indirect + these two
# == 1.0 exactly) and for auditing.
DIAGNOSTIC_SHARE_NUMERATOR_FIELDS: dict[str, str] = {
    "noncompetitive_share": "noncomp_accepted",
    "fima_share": "fima_noncomp_accepted",
}

PUBLIC_ACCEPTED_COL = "public_accepted_amount"
FED_RESERVE_ADDON_COL = "fed_reserve_addon_amount"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "comp_accepted",
    "noncomp_accepted",
    "fima_noncomp_accepted",
    "comp_tendered",
    "total_accepted",
    "total_tendered",
    "bid_to_cover_ratio",
    "soma_accepted",
    *SHARE_NUMERATOR_FIELDS.values(),
)


def _missing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def add_public_accepted_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Add `public_accepted_amount` (the correct share/bid-to-cover
    denominator: `comp_accepted + noncomp_accepted + fima_noncomp_accepted`,
    matching Treasury's own published "Subtotal") and
    `fed_reserve_addon_amount` (`soma_accepted` -- the excluded Fed
    rollover, kept for transparency/audit, never used as a numerator or
    denominator anywhere).

    Both are `pd.NA` whenever any required input is missing -- this
    function never guesses a value for an auction whose results aren't
    in yet.
    """
    missing = _missing_columns(df)
    if missing:
        raise KeyError(f"add_public_accepted_amount: missing required columns {missing}")

    out = df.copy()
    out[PUBLIC_ACCEPTED_COL] = out["comp_accepted"] + out["noncomp_accepted"] + out["fima_noncomp_accepted"]
    out[FED_RESERVE_ADDON_COL] = out["soma_accepted"]
    return out


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Nullable-float division that maps a zero denominator to missing
    (rather than +/-inf) instead of dividing by it. NA in either input
    already propagates to NA through ordinary arithmetic on pandas'
    nullable Float64 dtype.
    """
    share = numerator / denominator
    return share.mask(denominator == 0)


def add_bidder_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three primary candidate bidder-category shares plus two
    diagnostic shares (`noncompetitive_share`, `fima_share`), all using
    `public_accepted_amount` as the shared, economically-appropriate
    denominator. See the module docstring for the formula and why.

    Requires `add_public_accepted_amount` to have been called already
    (or calls it here if `public_accepted_amount` is absent).
    """
    out = df if PUBLIC_ACCEPTED_COL in df.columns else add_public_accepted_amount(df)
    out = out.copy()
    denominator = out[PUBLIC_ACCEPTED_COL]
    for share_col, numerator_col in {**SHARE_NUMERATOR_FIELDS, **DIAGNOSTIC_SHARE_NUMERATOR_FIELDS}.items():
        out[share_col] = _safe_share(out[numerator_col], denominator)
    return out


def add_bid_to_cover(df: pd.DataFrame) -> pd.DataFrame:
    """Add `bid_to_cover_calculated` (the from-scratch reconstruction,
    matching Treasury's own "Subtotal tendered / Subtotal accepted"
    methodology exactly) alongside the untouched API field
    `bid_to_cover_ratio`, plus `bid_to_cover_reconciliation_diff` (API
    minus calculated) so the reconciliation is auditable row-by-row.
    """
    missing = _missing_columns(df)
    if missing:
        raise KeyError(f"add_bid_to_cover: missing required columns {missing}")

    out = df if PUBLIC_ACCEPTED_COL in df.columns else add_public_accepted_amount(df)
    out = out.copy()
    public_tendered = out["comp_tendered"] + out["noncomp_accepted"] + out["fima_noncomp_accepted"]
    out["bid_to_cover_calculated"] = _safe_share(public_tendered, out[PUBLIC_ACCEPTED_COL])
    out["bid_to_cover_reconciliation_diff"] = out["bid_to_cover_ratio"] - out["bid_to_cover_calculated"]
    return out


def add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: add the denominator, all bidder shares, and
    the bid-to-cover reconciliation in one call.
    """
    out = add_public_accepted_amount(df)
    out = add_bidder_shares(out)
    out = add_bid_to_cover(out)
    return out


def reconcile_bid_to_cover(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize how well the API's `bid_to_cover_ratio` matches this
    module's from-scratch calculation, rounded to 2 decimal places (the
    API's own display precision). Used by the profile report and by
    tests -- every number here must come from actually running this
    against real data, per `docs/project_rules.md`'s "no fabricated results" rule.
    """
    has_targets = "bid_to_cover_calculated" in df.columns
    working = df if has_targets else add_bid_to_cover(df)
    valid = working.dropna(subset=["bid_to_cover_ratio", "bid_to_cover_calculated"])
    if valid.empty:
        return {"n_compared": 0}

    rounded_calc = valid["bid_to_cover_calculated"].astype("Float64").round(2)
    diff = (valid["bid_to_cover_ratio"] - rounded_calc).abs()
    diff_float = diff.astype("float64")
    return {
        "n_compared": len(valid),
        "max_abs_diff": float(diff_float.max()),
        "mean_abs_diff": float(diff_float.mean()),
        "n_within_0_01": int((diff_float <= 0.01).sum()),
        "n_within_0_02": int((diff_float <= 0.02).sum()),
        "n_over_0_02": int((diff_float > 0.02).sum()),
    }
