"""Phase 6, Section 8: the required audit of
`treasury_auction_stress.features.dealer_absorption.add_regime_feature`
before any Phase 6 model could use it.

## The concrete risk

`add_regime_feature` sorts by `auction_date` and computes, within each
tenor, `shift(1).rolling(window=8).mean()` over `primary_dealer_share`
-- i.e. every one of the 8 most recent same-tenor auctions can
contribute to a given row's regime value, not merely the single
immediate predecessor. This is proven leakage-safe in Phase 2's own
sense: each row's value only ever depends on strictly-earlier rows in
`auction_date` order, so appending future rows never changes an
existing row's value (`tests/test_dealer_absorption.py`). **That
"no-future-leakage" property is real and still holds** -- it is a
narrower, different guarantee than what a Phase 6 PREDICTOR needs: a
predictor for auction *i* may only depend on results that were
**publicly, safely available by auction i's own forecast cutoff** --
not merely "occurred on an earlier calendar date." A reader should not
take "leakage-safe" (as used in `dealer_absorption.py`'s own docstring
and in `docs/target_specification.md`) to mean "safe to reuse directly
as a same-day/cutoff-respecting Phase 6 predictor" -- this module
exists precisely because those are different claims, and this project
found real, verified cases where the second one fails.

This module checks **every one of the up to 8 lookback slots**
`add_regime_feature`'s rolling window can draw from, at **both**
prediction cutoffs (announcement and pre-auction) -- not merely the
single immediate (`shift(1)`) predecessor at the announcement cutoff,
which was this audit's original, narrower scope. Checking the full
window at both cutoffs found MORE real violations than the
single-predecessor check alone (see module-level constants and
`summarize_regime_feature_risk`'s docstring for the exact, current
counts) -- concrete evidence for the protocol decision in
`configs/phase_6_evaluation.yml` (`dealer_absorption_surprise_treatment`)
to defer any Dealer Absorption Surprise / stress-event modeling to
Phase 7 rather than build a hasty, possibly-leaky version now.

## A real, verified example (not a hypothetical)

Two 7-Year auctions were BOTH announced on 2013-08-22
(`announcemt_date`): a reopening (CUSIP `912828RE2`, `auction_date`
2013-08-28) and a new issue (CUSIP `912828VV9`, `auction_date`
2013-08-29), one calendar day apart. `add_regime_feature`'s rolling
window would treat the reopening's actual `primary_dealer_share` as
part of the "8 most recent same-tenor auctions" feeding the new
issue's regime value (at lag 1, the most heavily-weighted slot). But
the reopening's own result only becomes safely available
(`evaluation.timing.add_result_safe_available_date`) on 2013-08-29 --
the new issue's OWN auction date, six days after the shared
2013-08-22 announcement cutoff both auctions actually need to stand
at, and still one day after even the LATER pre-auction cutoff
(2013-08-28). A naive reuse of `add_regime_feature` as a Phase 6
predictor would let the new issue's training-fold feature depend on an
outcome that was not yet publicly knowable at that auction's own
forecast cutoff, under either cutoff view.

A second, independent real case this fuller audit newly found (not
visible to the original single-predecessor check): CUSIP `9128286L9`
(7-Year, auctioned 2019-03-28) has TWO violating lookback slots at the
announcement cutoff simultaneously -- lag 1 (against `912828W71`,
auctioned 2019-03-27) AND lag 2 (against `912828C57`, auctioned
2019-03-26) -- because three 7-Year auctions were announced in rapid
succession that week. The lag-2 violation resolves by the (later)
pre-auction cutoff, but the lag-1 one does not.
"""

from __future__ import annotations

import pandas as pd

from treasury_auction_stress.evaluation.timing import (
    RESULT_SAFE_AVAILABLE_DATE_COL,
    add_result_safe_available_date,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)

# Matches dealer_absorption.REGIME_WINDOW_AUCTIONS exactly -- this
# audit must check the SAME window size the real feature uses, not an
# arbitrary smaller one.
REGIME_WINDOW_AUCTIONS = 8

TENOR_COL = "tenor"
DATE_COL = "auction_date"

CUTOFF_COLUMNS: tuple[str, ...] = (ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL)


def find_same_tenor_availability_violations(
    sample: pd.DataFrame, *, window: int = REGIME_WINDOW_AUCTIONS, cutoff_columns: tuple[str, ...] = CUTOFF_COLUMNS
) -> pd.DataFrame:
    """For every tenor, sort by `auction_date` and check EVERY lag from
    1 to `window` (i.e. every lookback slot `add_regime_feature`'s
    `rolling(window=window)` could draw from) against EVERY cutoff in
    `cutoff_columns`. A negative gap at (row, lag, cutoff) means: if
    that lagged same-tenor auction's result were naively folded into
    `add_regime_feature`'s rolling mean and used as a Phase 6 predictor
    for this row under this cutoff, it would reference a result that
    was NOT yet safely available at this row's own forecast cutoff.

    Returns one row per violating (auction, lag, cutoff) combination --
    a single auction can appear more than once (different lags and/or
    different cutoffs). An empty frame means no such case exists in
    `sample` for any checked lag or cutoff.
    """
    working = sample.copy()
    if RESULT_SAFE_AVAILABLE_DATE_COL not in working.columns:
        working = add_result_safe_available_date(working, date_col=DATE_COL)

    violations = []
    for cutoff_col in cutoff_columns:
        for _, group in working.groupby(TENOR_COL):
            group = group.sort_values(DATE_COL).reset_index(drop=True)
            for lag in range(1, window + 1):
                prev_safe = group[RESULT_SAFE_AVAILABLE_DATE_COL].shift(lag)
                prev_auction_date = group[DATE_COL].shift(lag)
                prev_cusip = (
                    group["cusip"].shift(lag) if "cusip" in group.columns else pd.Series(pd.NA, index=group.index)
                )
                gap_days = (group[cutoff_col] - prev_safe).dt.days
                flagged = group.loc[prev_safe.notna() & (gap_days < 0)].copy()
                if flagged.empty:
                    continue
                flagged["cutoff_view_column"] = cutoff_col
                flagged["lag"] = lag
                flagged["gap_days"] = gap_days.loc[flagged.index]
                flagged["previous_same_tenor_auction_date"] = prev_auction_date.loc[flagged.index]
                flagged["previous_same_tenor_cusip"] = prev_cusip.loc[flagged.index]
                violations.append(flagged)

    cols = [
        "cusip",
        TENOR_COL,
        DATE_COL,
        "announcemt_date",
        "cutoff_view_column",
        "lag",
        "previous_same_tenor_auction_date",
        "previous_same_tenor_cusip",
        "gap_days",
    ]
    if not violations:
        return pd.DataFrame(columns=cols)

    out = pd.concat(violations)
    return (
        out[[c for c in cols if c in out.columns]]
        .sort_values([DATE_COL, "cutoff_view_column", "lag"])
        .reset_index(drop=True)
    )


def summarize_regime_feature_risk(sample: pd.DataFrame, *, window: int = REGIME_WINDOW_AUCTIONS) -> dict:
    """A small, code-derived summary for `artifacts/phase_6_evaluation_
    protocol.md` and `artifacts/phase_6_leakage_audit.md`: how many real
    (auction, lag, cutoff) combinations in the sample would have had a
    naively-reused regime feature reference a not-yet-safely-available
    prior outcome, checking the FULL `window`-sized lookback at BOTH
    prediction cutoffs -- not merely the immediate predecessor at one
    cutoff (this audit's original, narrower scope).
    """
    violations = find_same_tenor_availability_violations(sample, window=window)
    by_cutoff = (
        violations.groupby("cutoff_view_column").size().to_dict() if not violations.empty else {}
    )
    by_lag = violations.groupby("lag").size().to_dict() if not violations.empty else {}
    return {
        "n_auctions_checked": len(sample),
        "window_checked": window,
        "cutoffs_checked": list(CUTOFF_COLUMNS),
        "n_total_violations": len(violations),
        "n_distinct_auctions_affected": int(violations["cusip"].nunique()) if not violations.empty else 0,
        "n_violations_by_cutoff": by_cutoff,
        "n_violations_by_lag": {int(k): int(v) for k, v in by_lag.items()},
        "min_gap_days": float(violations["gap_days"].min()) if not violations.empty else None,
        "example_violation": (
            violations.iloc[0][
                ["cusip", "tenor", "auction_date", "cutoff_view_column", "lag", "previous_same_tenor_cusip", "gap_days"]
            ].to_dict()
            if not violations.empty
            else None
        ),
    }
