"""Phase 7: a generic, reusable walk-forward (cross-fitted) construction
inside a single outer fold's own training pool.

`treasury_auction_stress.features.dealer_absorption.
compute_walk_forward_surprise` already implements this pattern once,
specifically for `DealerAbsorptionSurpriseModel`, over the FULL eligible
history (for Phase 2's descriptive purposes). Phase 7 needs the exact
same walk-forward discipline in two more places -- generating
out-of-fold residuals to calibrate quantile forecasts, and generating
out-of-fold Dealer Absorption Surprise labels to train the stress-event
classifier -- both scoped to a single OUTER fold's own training pool,
never the full history. `walk_forward_transform` below is that shared,
generic primitive so neither caller re-derives the same logic.

## Why this exists (the in-sample-residual-understatement problem)

Fitting a model once on an outer fold's full training pool and then
scoring that SAME pool's own residuals understates residual variance:
whatever the model was fit to minimize (e.g. OLS residual sum of
squares) is necessarily smaller in-sample than it would be out-of-
sample. Both Phase 7 consumers of this function need an honest,
OUT-OF-FOLD residual/label distribution from *within* the training
pool -- never the final model's own in-sample fit -- to calibrate a
threshold or a quantile without peeking at the outer test year at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from treasury_auction_stress.evaluation.timing import (
    RESULT_SAFE_AVAILABLE_DATE_COL,
    annual_fold_years,
    build_fold,
)
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL


def walk_forward_transform(
    pool: pd.DataFrame,
    *,
    fit_fn: Callable[[pd.DataFrame], Any],
    transform_fn: Callable[[Any, pd.DataFrame], pd.DataFrame],
    date_col: str = "auction_date",
    announcement_cutoff_col: str = ANNOUNCEMENT_CUTOFF_COL,
    safe_date_col: str = RESULT_SAFE_AVAILABLE_DATE_COL,
) -> pd.DataFrame:
    """Walk forward one calendar year at a time within `pool` (typically
    an outer fold's own training pool, never the outer test year):
    for every distinct year present except the EARLIEST, build a fold
    via the exact same `treasury_auction_stress.evaluation.timing.
    build_fold` function every outer/inner fold in this project already
    uses (so this walk-forward also respects the result-availability
    rule and same-day auction grouping), fit `fit_fn` on that fold's
    own training rows (strictly-prior years, availability-filtered),
    and apply `transform_fn(model, fold.test)` to that year's own rows.

    Returns the concatenation of every transformed inner year. Rows
    belonging to `pool`'s own earliest calendar year are NEVER included
    in the result (there is no strictly-prior inner fold to fit on
    yet) -- callers must treat this as "no cross-fitted value exists
    for these rows," never silently default them to zero or to the
    fold's own unconditional mean.

    Returns an empty frame (0 rows, `pool`'s own columns) if `pool`
    spans fewer than 2 distinct years, or if every year's own fold
    happens to be unusable (fit_fn raising is NOT caught here --
    callers whose `fit_fn` can legitimately fail on a small inner
    fold must handle that themselves; this function only skips a year
    whose fold construction itself yields an empty train or test set).
    """
    years = annual_fold_years(pool, date_col=date_col)
    if len(years) < 2:
        return pool.iloc[0:0].copy()

    frames: list[pd.DataFrame] = []
    for year in years[1:]:
        fold = build_fold(
            pool,
            test_year=year,
            date_col=date_col,
            announcement_cutoff_col=announcement_cutoff_col,
            safe_date_col=safe_date_col,
        )
        if fold.train.empty or fold.test.empty:
            continue
        model = fit_fn(fold.train)
        frames.append(transform_fn(model, fold.test))

    if not frames:
        return pool.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=False)
