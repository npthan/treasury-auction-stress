"""Phase 6: the pre-registered baseline ladder (models 1-4 of
`configs/phase_6_evaluation.yml`'s `models` list). Every baseline is a
`fit(train_df, target_col) -> self` / `predict(test_df) -> pd.Series`
object, fit once per fold on that fold's own eligible training pool,
frozen for the whole test year -- never re-estimated using a test-year
outcome (see `treasury_auction_stress.evaluation.timing`).

None of these baselines look at any predictor beyond `tenor` and
`is_reopening` (both known exactly at announcement time, for both
cutoff views) -- they exist to answer "how much does a fitted
regression actually add over a trivial, already-known-at-announcement
grouping," not to be sophisticated in their own right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TENOR_COL = "tenor"
REOPENING_COL = "is_reopening"
DATE_COL = "auction_date"
AUCTION_KEY_COL = "auction_key"


@dataclass
class GlobalMeanBaseline:
    """Model 1: a single constant, the training pool's own mean of the
    target. No fallback hierarchy needed -- there is only one group.
    """

    target_col: str = "primary_dealer_share"
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> GlobalMeanBaseline:
        clean = train_df[self.target_col].dropna()
        if clean.empty:
            raise ValueError("GlobalMeanBaseline.fit: no non-missing training targets")
        self.global_mean_ = float(clean.mean())
        self.n_train_ = len(clean)
        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("GlobalMeanBaseline.predict called before fit")
        return pd.Series(self.global_mean_, index=test_df.index)


@dataclass
class TenorMeanBaseline:
    """Model 2: one constant per tenor. A tenor absent from training
    (structurally impossible for the 6 pre-2020 tenors, but real for
    20-Year in any fold before its own history exists) falls back to
    the global training mean.
    """

    target_col: str = "primary_dealer_share"
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    tenor_means_: dict = field(default_factory=dict, init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> TenorMeanBaseline:
        clean = train_df.dropna(subset=[self.target_col])
        if clean.empty:
            raise ValueError("TenorMeanBaseline.fit: no non-missing training targets")
        self.global_mean_ = float(clean[self.target_col].mean())
        self.tenor_means_ = clean.groupby(TENOR_COL)[self.target_col].mean().to_dict()
        self.n_train_ = len(clean)
        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("TenorMeanBaseline.predict called before fit")
        return test_df[TENOR_COL].map(self.tenor_means_).fillna(self.global_mean_)


@dataclass
class TenorReopeningMeanBaseline:
    """Model 3: one constant per (tenor, is_reopening) pair. Pre-declared
    (`configs/phase_6_evaluation.yml`, `is_strong_baseline: true`) as
    THE strong historical baseline for benchmark-relative R2.

    Fallback hierarchy for a sparse/unseen group, applied independently
    for every test row: (tenor, is_reopening) group mean -> tenor-only
    mean -> global training mean -- whichever is the first level with
    at least 1 training-eligible observation.
    """

    target_col: str = "primary_dealer_share"
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    tenor_means_: dict = field(default_factory=dict, init=False, repr=False)
    group_means_: dict = field(default_factory=dict, init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> TenorReopeningMeanBaseline:
        clean = train_df.dropna(subset=[self.target_col])
        if clean.empty:
            raise ValueError("TenorReopeningMeanBaseline.fit: no non-missing training targets")
        self.global_mean_ = float(clean[self.target_col].mean())
        self.tenor_means_ = clean.groupby(TENOR_COL)[self.target_col].mean().to_dict()
        self.group_means_ = clean.groupby([TENOR_COL, REOPENING_COL])[self.target_col].mean().to_dict()
        self.n_train_ = len(clean)
        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("TenorReopeningMeanBaseline.predict called before fit")

        def _one(row: pd.Series) -> float:
            key = (row[TENOR_COL], bool(row[REOPENING_COL]))
            if key in self.group_means_:
                return self.group_means_[key]
            if row[TENOR_COL] in self.tenor_means_:
                return self.tenor_means_[row[TENOR_COL]]
            return self.global_mean_

        return test_df.apply(_one, axis=1)


DEFAULT_RECENT_HISTORY_LOOKBACK = 8


@dataclass
class RecentHistoryBaseline:
    """Model 4: the mean of the `n_lookback` most recent training-
    eligible same-tenor outcomes, as of the fold's own fit origin.
    Computed once at `fit` time and frozen for the whole test year --
    this is a CONSTANT-per-tenor forecast for the fold, not a
    row-by-row rolling computation re-evaluated per test auction
    (`configs/phase_6_evaluation.yml` requires every model be frozen
    for its test year; using a genuinely rolling window that keeps
    advancing through the test year would violate that).

    Fallback: if fewer than `n_lookback` eligible same-tenor training
    rows exist, use however many exist; if zero exist for that tenor,
    fall back to the global training mean.

    ## Deterministic tie-break (acceptance-review fix)

    Sorting by `DATE_COL` alone is not sufficient: if two same-tenor
    auctions ever shared the exact same `auction_date`, `pandas.sort_
    values`'s stability means the tied rows keep whatever RELATIVE
    order they happened to arrive in on the input dataframe -- so
    `.tail(n_lookback)` could silently select a different subset of
    tied rows (and therefore a different forecast) depending on
    upstream row order alone, with no error. This is the same class of
    defect the Phase 5 acceptance review found and fixed in
    `cftc_join.py` (and defensively in three other join modules) for
    tied `publication_safe_available_date` values -- see
    the Phase 5 acceptance review, Issue 2. Verified directly
    (`tests/test_phase6_baselines.py::
    test_recent_history_baseline_is_order_invariant_even_with_a_same_day_tie`):
    swapping which of two identically-dated rows is fed in first to
    `fit` flips the resulting forecast from 0.9 to 0.3 without this fix.
    No same-tenor, same-`auction_date` auction exists anywhere in this
    project's real 2010-2026 sample (verified:
    `ann.groupby(["tenor","auction_date"]).size()` has zero groups
    above 1), so this defect never actually changed any previously
    published Phase 6 number -- but nothing in the pre-fix code
    *guaranteed* that, and this project's own standing discipline
    (`docs/point_in_time_rules.md`) requires a real guarantee, not a
    coincidence. The fix sorts by `[DATE_COL, tie_break_col]`, where
    `tie_break_col` (default `"auction_key"`, always present on every
    real fold-training frame this project builds) is a stable, unique,
    content-derived key -- never the dataframe's own positional index,
    which would just relocate the same non-determinism one level down.
    """

    target_col: str = "primary_dealer_share"
    n_lookback: int = DEFAULT_RECENT_HISTORY_LOOKBACK
    tie_break_col: str = AUCTION_KEY_COL
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    tenor_recent_means_: dict = field(default_factory=dict, init=False, repr=False)
    tenor_n_used_: dict = field(default_factory=dict, init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> RecentHistoryBaseline:
        if self.tie_break_col not in train_df.columns:
            raise KeyError(
                f"RecentHistoryBaseline.fit: deterministic tie-breaking requires column "
                f"{self.tie_break_col!r} (a stable, unique per-auction key); it is missing from train_df"
            )
        clean = train_df.dropna(subset=[self.target_col]).sort_values([DATE_COL, self.tie_break_col])
        if clean.empty:
            raise ValueError("RecentHistoryBaseline.fit: no non-missing training targets")
        self.global_mean_ = float(clean[self.target_col].mean())
        self.n_train_ = len(clean)
        for tenor, group in clean.groupby(TENOR_COL):
            recent = group.tail(self.n_lookback)
            self.tenor_recent_means_[tenor] = float(recent[self.target_col].mean())
            self.tenor_n_used_[tenor] = len(recent)
        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("RecentHistoryBaseline.predict called before fit")
        return test_df[TENOR_COL].map(self.tenor_recent_means_).fillna(self.global_mean_)
