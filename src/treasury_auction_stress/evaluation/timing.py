"""Phase 6: the chronological, leakage-safe fold-construction core.

Every other Phase 6 module (baselines, linear models, the evaluation
driver) builds folds through `build_fold` below -- never by re-deriving
a train/test split ad hoc. See `configs/phase_6_evaluation.yml`'s
`label_availability` and `fold_schedule` sections for the frozen
protocol this module implements.

## `result_safe_available_date`: a disclosed, conservative proxy

The Treasury Fiscal Data auctions API (this project's only auction
source) does not publish a verified RESULT release timestamp -- only
`auction_date` and `announcemt_date` exist on the schema (see
`docs/point_in_time_rules.md`, `docs/data_dictionary.md`). No Phase
5 artifact carries a result-availability column; this module is the
first place in the project that constructs one, purely for Phase 6's
own training-eligibility rule.

`result_safe_available_date = next_full_business_day_after(auction_date)`,
reusing the exact same shared safe-availability rule
(`time_utils.next_full_business_day_after`) every other Phase 3/4/5
publication-timing decision already uses. This is a **conservative**
proxy: Treasury's own results press releases are dated the auction date
itself (results are public within hours of the auction closing), so
this proxy can only make a training row look *less* available than it
really was -- never more. No exact intraday release time is invented.

## Why "auction_date < fold year" alone is not sufficient (and why it is safe here)

`docs/project_rules.md` explicitly warns against assuming `auction_date < fold year`
proves a result was safely available by the fit origin. This module
never makes that assumption directly -- it always additionally applies
`result_safe_available_date <= fold_fit_origin`. It is worth writing
down *why* the two conditions turn out to agree almost everywhere in
this project's real data, so a future reader does not mistake the
extra filter for dead code: a test-year auction's own announcement
cutoff is, by construction, always `>= fold_fit_origin` (the fold's fit
origin is defined as the *minimum* over exactly the test year's own
announcement cutoffs). Its own auction necessarily happens strictly
after its own announcement, and its own result becomes safely available
strictly after its own auction -- so a same-year test auction's result
can never be safely available at or before `fold_fit_origin`. The extra
filter therefore does no work in the ordinary case, but it is not
redundant in general: it is what correctly excludes a handful of
**prior-year** auctions whose announcement (and hence result) happened
very close to a fold's earliest test-year announcement (e.g. a
December auction announced days before the following January's first
auction) -- exactly the "same relevant announcement date" boundary
case `docs/point_in_time_rules.md` and the Phase 6 task specification
call out. `tests/test_phase6_timing.py` proves both properties directly
against constructed fixtures, including one modeled on a real batched
December/January announcement pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from treasury_auction_stress.data.time_utils import (
    next_full_business_day_after,
    us_federal_holidays,
)
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL

RESULT_SAFE_AVAILABLE_DATE_COL = "result_safe_available_date"


def add_result_safe_available_date(df: pd.DataFrame, *, date_col: str = "auction_date") -> pd.DataFrame:
    """Add `result_safe_available_date` -- see module docstring for the
    exact rule and its disclosed limitation. Never mutates `df` in
    place.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col]).dt.normalize()
    holidays = us_federal_holidays(
        start=(dates.min() - pd.Timedelta(days=14)).isoformat(),
        end=(dates.max() + pd.Timedelta(days=14)).isoformat(),
    )
    out[RESULT_SAFE_AVAILABLE_DATE_COL] = dates.map(lambda d: next_full_business_day_after(d, holidays))
    return out


@dataclass(frozen=True)
class Fold:
    """One chronological fold: a frozen fit origin, a training pool
    (already filtered to eligible rows), and a test set. `test_year` is
    an `int` for an ordinary annual fold; the inner-CV holdout fold
    (built by `build_inner_holdout`) reuses this same structure.
    """

    test_year: int
    fit_origin: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame
    n_train_dropped_for_year_boundary: int


def build_fold(
    pool: pd.DataFrame,
    *,
    test_year: int,
    date_col: str = "auction_date",
    announcement_cutoff_col: str = ANNOUNCEMENT_CUTOFF_COL,
    safe_date_col: str = RESULT_SAFE_AVAILABLE_DATE_COL,
) -> Fold:
    """Build one annual, expanding-window fold from `pool` (which must
    already carry `safe_date_col`, e.g. via `add_result_safe_available_date`).

    - `test` = every row of `pool` with `date_col` in `test_year`.
    - `fit_origin` = the MINIMUM `announcement_cutoff_col` over `test`
      -- shared by both cutoff views (see module docstring and
      `configs/phase_6_evaluation.yml`).
    - `train` = every row with `date_col` strictly before `test_year`
      (guarantees train/test disjointness for an expanding-window
      design) AND `safe_date_col <= fit_origin`.

    Raises `ValueError` if `test_year` has no rows in `pool` at all --
    an empty test set is always a caller bug (a typo'd year, or a
    year outside the sample's real coverage), never a legitimate fold.
    """
    years = pd.to_datetime(pool[date_col]).dt.year
    test = pool.loc[years == test_year].copy()
    if test.empty:
        raise ValueError(f"build_fold: no rows in pool for test_year={test_year}")

    fit_origin = pd.Timestamp(test[announcement_cutoff_col].min())

    prior_years_pool = pool.loc[years < test_year]
    train = prior_years_pool.loc[prior_years_pool[safe_date_col] <= fit_origin].copy()
    n_dropped = len(prior_years_pool) - len(train)

    return Fold(
        test_year=test_year,
        fit_origin=fit_origin,
        train=train,
        test=test,
        n_train_dropped_for_year_boundary=n_dropped,
    )


def assert_fold_is_leakage_safe(
    fold: Fold,
    *,
    date_col: str = "auction_date",
    safe_date_col: str = RESULT_SAFE_AVAILABLE_DATE_COL,
) -> None:
    """A standalone, reusable adversarial check (used by both
    `tests/test_phase6_timing.py`'s regression tests and its
    deliberately-leaky negative-control fixture): raises
    `AssertionError` if any training row's `safe_date_col` is after
    `fold.fit_origin`, or if any auction key appears in both `fold.train`
    and `fold.test`. A correctly-built `Fold` (via `build_fold`) always
    passes; this function's whole purpose is to also correctly REJECT a
    deliberately-broken fold object, proving the check has teeth.
    """
    from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL

    if not fold.train.empty:
        late = fold.train[safe_date_col] > fold.fit_origin
        if late.any():
            bad = sorted(fold.train.loc[late, AUCTION_KEY_COL])[:10] if AUCTION_KEY_COL in fold.train.columns else []
            raise AssertionError(
                f"assert_fold_is_leakage_safe: {int(late.sum())} training row(s) have "
                f"{safe_date_col} after fit_origin={fold.fit_origin} (e.g. {bad})"
            )

    if AUCTION_KEY_COL in fold.train.columns and AUCTION_KEY_COL in fold.test.columns:
        overlap = set(fold.train[AUCTION_KEY_COL]) & set(fold.test[AUCTION_KEY_COL])
        if overlap:
            raise AssertionError(f"assert_fold_is_leakage_safe: train/test auction_key overlap: {sorted(overlap)[:10]}")


def annual_fold_years(pool: pd.DataFrame, *, date_col: str = "auction_date") -> list[int]:
    """All distinct calendar years present in `pool[date_col]`, sorted."""
    return sorted(pd.to_datetime(pool[date_col]).dt.year.unique().tolist())


@dataclass(frozen=True)
class InnerHoldout:
    """The single chronological inner-validation split used for
    hyperparameter selection, nested inside one outer fold's own
    training pool. See `configs/phase_6_evaluation.yml`'s
    `hyperparameter_selection` section.
    """

    usable: bool
    inner_train: pd.DataFrame
    inner_val: pd.DataFrame
    reason_if_unusable: str | None


MIN_INNER_CV_YEARS = 2
MIN_INNER_CV_TRAIN_ROWS = 50


def build_inner_holdout(
    outer_train_pool: pd.DataFrame,
    *,
    date_col: str = "auction_date",
    announcement_cutoff_col: str = ANNOUNCEMENT_CUTOFF_COL,
    safe_date_col: str = RESULT_SAFE_AVAILABLE_DATE_COL,
) -> InnerHoldout:
    """The outer training pool's own most recent complete calendar year
    becomes the inner validation set; every strictly earlier year is
    the inner training set -- built via the exact same `build_fold`
    logic as an outer fold (so the inner split also respects result
    availability and same-day auction grouping), never a separate,
    ad hoc split.

    Returns `usable=False` (with a reason) if the outer training pool
    does not have at least `MIN_INNER_CV_YEARS` distinct years, or if
    the resulting inner-training pool has fewer than
    `MIN_INNER_CV_TRAIN_ROWS` rows -- callers must fall back to the
    protocol's pre-declared fixed hyperparameters in that case (see
    `configs/phase_6_evaluation.yml`'s `hyperparameter_selection.
    fallback_if_insufficient_history`).
    """
    years = annual_fold_years(outer_train_pool, date_col=date_col)
    if len(years) < MIN_INNER_CV_YEARS:
        return InnerHoldout(
            usable=False,
            inner_train=outer_train_pool.iloc[0:0],
            inner_val=outer_train_pool.iloc[0:0],
            reason_if_unusable=f"outer training pool spans only {len(years)} distinct year(s), need >= {MIN_INNER_CV_YEARS}",
        )

    inner_val_year = years[-1]
    inner_fold = build_fold(
        outer_train_pool,
        test_year=inner_val_year,
        date_col=date_col,
        announcement_cutoff_col=announcement_cutoff_col,
        safe_date_col=safe_date_col,
    )
    if len(inner_fold.train) < MIN_INNER_CV_TRAIN_ROWS:
        return InnerHoldout(
            usable=False,
            inner_train=inner_fold.train,
            inner_val=inner_fold.test,
            reason_if_unusable=(
                f"inner-training pool has only {len(inner_fold.train)} row(s), need >= {MIN_INNER_CV_TRAIN_ROWS}"
            ),
        )
    return InnerHoldout(
        usable=True, inner_train=inner_fold.train, inner_val=inner_fold.test, reason_if_unusable=None
    )
