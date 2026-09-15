"""Dealer Absorption Surprise: a leakage-safe transformer, and a
provisional stress-event label built on top of it.

## Concept

Dealer Absorption Surprise is the residual of `primary_dealer_share`
(see `treasury_auction_stress.features.targets`) after removing the
part of it that is predictable from known structural factors: tenor,
reopening status, offering size, and a slowly-changing historical
regime. See `docs/target_specification.md` for the economic argument
for why a high dealer share is not automatically evidence of a weak
auction, and why this residualization matters.

## Why this must be a fit/transform object, not a function

If the tenor/offering/regime -> expected-share relationship were
estimated once over the whole dataset, an auction from 2012 would have
its "expected share" computed using a relationship that was partly
learned from auctions in 2024 -- a real-time forecaster in 2012 could
never have seen that. `DealerAbsorptionSurpriseModel` therefore
separates `fit` (learns parameters from a training-fold dataframe
only) from `transform` (applies the already-frozen parameters to any
dataframe, train or test). Nothing in `transform` re-estimates
anything from its input -- see the tests in `tests/test_dealer_absorption.py`
for a direct proof that adding future rows to a `transform` call never
changes an already-fitted model's output for existing rows.

## Model: interpretable by design, not a starting point to be judged as final

Per `docs/target_specification.md`, this is deliberately simple:

    expected_share(auction) =
        group_mean[tenor, is_reopening]                      (learned)
      + offering_slope * (log(offering_amt) - offering_center)   (learned, 1 coefficient)
      + regime_slope   * (regime_feature - regime_center)        (learned, 1 coefficient)

`group_mean` captures "2-year vs. 30-year auctions have systematically
different typical dealer shares" and "new issues vs. reopenings behave
differently" in one term. `offering_slope` captures "a larger offering
size, for a given tenor/reopening bucket, tends to move dealer
take-down up or down" as a single, sign-interpretable coefficient
(fit by ordinary least squares on the group-mean residual, using
`numpy.linalg.lstsq` -- no extra ML dependency, no regularization to
tune). `regime_slope` does the same for the slowly-changing regime
feature (see `add_regime_feature` below). This is intentionally not a
gradient-boosted tree or anything else that would obscure why a given
auction's expected share came out the way it did; a more sophisticated
residualization model is explicit future-phase work, not this one.

## The regime feature is leakage-safe by construction, not by fitting

`add_regime_feature` computes, for every auction, the trailing mean of
`primary_dealer_share` over the same tenor's preceding auctions,
**excluding the current auction itself** (`shift(1)` before
`rolling(...)`) -- the same rule
`docs/point_in_time_rules.md` states generally ("Rolling statistics
must exclude the target auction itself"). Because each row's value
only ever depends on strictly-earlier rows in sorted order, it is
already point-in-time-safe for any row the moment it is computed, and
computing it once over the full eligible history (rather than
separately per fold) does **not** leak information forward: a later
auction's presence in the same dataframe can never change an earlier
auction's trailing mean. `tests/test_dealer_absorption.py` proves this
directly. What *does* still need to be fold-scoped is the
`DealerAbsorptionSurpriseModel` fit itself (the group means and the two
slopes), because those parameters are estimated by looking at many
rows at once, not just a single row's own past.

**Acceptance-review caveat (Phase 6): "leakage-safe" here is a
narrower claim than "safe to use directly as a real-time predictor."**
The property proven above -- no future row's presence changes an
earlier row's own computed value -- says nothing about whether an
EARLIER auction's own result was actually publicly, safely available
by the CURRENT auction's own forecast cutoff. "Earlier in `auction_date`
order" is not automatically the same thing as "already public
knowledge by this auction's own cutoff," particularly for same-day or
closely-spaced same-tenor auctions and the date-level uncertainty in
exactly when a result becomes public (see
`treasury_auction_stress.evaluation.timing`'s
`result_safe_available_date` proxy). `treasury_auction_stress.
evaluation.dealer_absorption_audit` checked this directly against this
project's real data and found real, verified cases (up to 34
(auction, lookback-slot, cutoff) violations across 18 distinct
auctions, checking the full 8-auction window at both prediction
cutoffs -- see `artifacts/phase_6_evaluation_protocol.md` for the exact,
current counts) where naively reusing this feature as a same-day/
cutoff-respecting Phase 6 predictor would violate that stronger
requirement. This is why Phase 6 does not use `add_regime_feature` or
`DealerAbsorptionSurpriseModel` as a predictor and defers any Dealer
Absorption Surprise modeling to Phase 7 -- see `configs/
phase_6_evaluation.yml`'s `dealer_absorption_surprise_treatment`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_WINDOW_AUCTIONS = 8
REGIME_FEATURE_COL = "dealer_share_regime_trailing_mean"


def add_regime_feature(
    df: pd.DataFrame,
    *,
    tenor_col: str = "tenor",
    value_col: str = "primary_dealer_share",
    date_col: str = "auction_date",
    window: int = REGIME_WINDOW_AUCTIONS,
) -> pd.DataFrame:
    """Add `dealer_share_regime_trailing_mean`: the trailing mean of
    `value_col` over up to `window` preceding same-tenor auctions,
    strictly excluding the current row. `pd.NA` for the first auction(s)
    of a tenor that has no preceding history yet (e.g. the very first
    20-year auction after its 2020 reintroduction) -- never backfilled
    or imputed here; `DealerAbsorptionSurpriseModel.fit`/`.transform`
    decide how to handle that missingness (see below).
    """
    out = df.sort_values(date_col).copy()
    out[REGIME_FEATURE_COL] = out.groupby(tenor_col)[value_col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return out


@dataclass
class DealerAbsorptionSurpriseModel:
    """Fit on a training-fold dataframe only; transform any dataframe
    using the frozen parameters learned at fit time.
    """

    tenor_col: str = "tenor"
    reopening_col: str = "is_reopening"
    offering_col: str = "offering_amt"
    regime_col: str = REGIME_FEATURE_COL
    target_col: str = "primary_dealer_share"

    is_fitted_: bool = field(default=False, init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    regime_fallback_: float = field(default=float("nan"), init=False, repr=False)
    group_means_: dict = field(default_factory=dict, init=False, repr=False)
    offering_center_: float = field(default=float("nan"), init=False, repr=False)
    regime_center_: float = field(default=float("nan"), init=False, repr=False)
    offering_slope_: float = field(default=0.0, init=False, repr=False)
    regime_slope_: float = field(default=0.0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> DealerAbsorptionSurpriseModel:
        train = train_df.dropna(subset=[self.target_col, self.offering_col]).copy()
        if train.empty:
            raise ValueError("DealerAbsorptionSurpriseModel.fit: no usable training rows")

        self.n_train_ = len(train)
        self.global_mean_ = float(train[self.target_col].mean())

        has_regime = self.regime_col in train.columns and train[self.regime_col].notna().any()
        self.regime_fallback_ = (
            float(train[self.regime_col].mean()) if has_regime else self.global_mean_
        )

        group_key = pd.Series(
            list(zip(train[self.tenor_col], train[self.reopening_col], strict=True)),
            index=train.index,
        )
        group_means = train[self.target_col].groupby(group_key).mean()
        self.group_means_ = group_means.to_dict()

        residual = train[self.target_col] - group_key.map(self.group_means_)
        log_offering = np.log(train[self.offering_col].astype("float64"))
        self.offering_center_ = float(log_offering.mean())

        regime = (
            train[self.regime_col].astype("float64")
            if self.regime_col in train.columns
            else pd.Series(np.nan, index=train.index)
        )
        regime = regime.fillna(self.regime_fallback_)
        self.regime_center_ = float(regime.mean())

        design = np.column_stack(
            [
                log_offering.to_numpy() - self.offering_center_,
                regime.to_numpy() - self.regime_center_,
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, residual.to_numpy(dtype="float64"), rcond=None)
        self.offering_slope_, self.regime_slope_ = float(coefficients[0]), float(coefficients[1])
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("DealerAbsorptionSurpriseModel.transform called before fit")

        out = df.copy()
        group_key = list(zip(out[self.tenor_col], out[self.reopening_col], strict=True))
        group_mean = pd.Series(
            [self.group_means_.get(key, self.global_mean_) for key in group_key],
            index=out.index,
        )
        log_offering = np.log(out[self.offering_col].astype("float64"))
        regime = (
            out[self.regime_col].astype("float64")
            if self.regime_col in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        regime = regime.fillna(self.regime_fallback_)

        expected = (
            group_mean
            + self.offering_slope_ * (log_offering - self.offering_center_)
            + self.regime_slope_ * (regime - self.regime_center_)
        )
        out["expected_dealer_share"] = expected
        out["dealer_absorption_surprise"] = out[self.target_col] - expected
        return out


def compute_walk_forward_surprise(
    df: pd.DataFrame, *, date_col: str = "auction_date", min_train_years: int = 1
) -> pd.DataFrame:
    """Descriptive, leakage-safe Dealer Absorption Surprise for the
    *whole* eligible history, built by walking forward one calendar
    year at a time: for each year, fit `DealerAbsorptionSurpriseModel`
    on every strictly-earlier year and transform only that year's
    auctions. This is not a predictive-model evaluation (Phase 2 does
    not train predictive models) -- it exists purely so the target's
    own distribution can be examined honestly, auction by auction,
    without any auction ever contributing to its own expected value or
    to another auction's expected value from the future.

    The first `min_train_years` calendar year(s) of history are
    excluded from the result entirely (there is no strictly-prior fold
    yet to fit on) -- this is expected and documented, not a bug.
    """
    working = df.sort_values(date_col).copy()
    years = sorted(working[date_col].dt.year.unique())
    folds = []
    for year in years[min_train_years:]:
        train = working.loc[working[date_col].dt.year < year]
        test = working.loc[working[date_col].dt.year == year]
        if train.empty or test.empty:
            continue
        model = DealerAbsorptionSurpriseModel().fit(train)
        folds.append(model.transform(test))
    if not folds:
        empty = working.copy()
        empty["expected_dealer_share"] = pd.NA
        empty["dealer_absorption_surprise"] = pd.NA
        return empty.iloc[0:0]
    return pd.concat(folds).sort_values(date_col)


@dataclass
class StressThreshold:
    """A percentile threshold fit on a training-fold surprise series
    only; `transform` never recomputes it from whatever is passed in.
    """

    percentile: float = 0.90

    threshold_: float = field(default=float("nan"), init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)
    is_fitted_: bool = field(default=False, init=False, repr=False)

    def fit(self, train_surprise: pd.Series) -> StressThreshold:
        clean = pd.Series(train_surprise).dropna()
        if clean.empty:
            raise ValueError("StressThreshold.fit: no non-missing training values")
        self.threshold_ = float(clean.quantile(self.percentile))
        self.n_train_ = len(clean)
        self.is_fitted_ = True
        return self

    def transform(self, surprise: pd.Series) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("StressThreshold.transform called before fit")
        s = pd.Series(surprise)
        flagged = s >= self.threshold_
        return flagged.mask(s.isna())


def describe_percentile_thresholds(
    train_surprise: pd.Series,
    percentiles: tuple[float, ...] = (0.75, 0.80, 0.90, 0.95),
) -> dict[str, dict]:
    """Descriptive comparison of several candidate stress-event
    percentile cutoffs, computed from a training-fold surprise series
    only. Purely descriptive -- this function never looks at any
    out-of-sample performance metric, per `docs/target_specification.md`
    ("the threshold... must be estimated using training data only" and
    must not be chosen using future model performance).
    """
    clean = pd.Series(train_surprise).dropna()
    result: dict[str, dict] = {}
    for pct in percentiles:
        threshold = StressThreshold(percentile=pct).fit(clean)
        n_flagged = int((clean >= threshold.threshold_).sum())
        result[f"p{round(pct * 100)}"] = {
            "percentile": pct,
            "threshold": threshold.threshold_,
            "n_train": threshold.n_train_,
            "n_flagged_in_training_window": n_flagged,
            "fraction_flagged_in_training_window": n_flagged / threshold.n_train_,
        }
    return result
