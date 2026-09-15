"""Phase 7: `quantile_recent_history_residual`
(`configs/phase_7_protocol.yml`'s `probabilistic.conservative_baseline_method`)
-- prediction intervals built around a frozen point forecast by adding
empirical quantiles of WALK-FORWARD (never in-sample) training
residuals.

Fitting a point model once on an outer fold's own training pool and
then measuring residuals on that SAME pool would understate the true
out-of-sample residual spread (the same in-sample-understatement
problem `treasury_auction_stress.evaluation.walk_forward` exists to
avoid). This module reuses that shared walk-forward primitive to
generate an honest, cross-fitted residual sample from the training
pool alone, then reads off empirical quantiles of that sample as
additive offsets around whatever point forecast the caller supplies
for the actual test year.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from treasury_auction_stress.evaluation.probabilistic_metrics import (
    quantile_column_name,
)
from treasury_auction_stress.evaluation.walk_forward import walk_forward_transform

TENOR_COL = "tenor"
RESIDUAL_COL = "_walk_forward_residual"

MIN_RESIDUALS_PER_TENOR = 30


@dataclass
class ResidualQuantileCalibrator:
    """`fit(train_pool, point_model_factory, point_predict_fn)` runs the
    walk-forward residual construction; `predict_quantiles(test_df,
    point_forecast)` adds the calibrated offsets to an already-computed
    point forecast for the test year (the SAME frozen point forecast
    the point-model comparison table reports, never a second, silently
    different one).
    """

    quantile_levels: tuple[float, ...]
    target_col: str = "primary_dealer_share"
    min_residuals_per_tenor: int = MIN_RESIDUALS_PER_TENOR

    is_fitted_: bool = field(default=False, init=False, repr=False)
    tenor_quantiles_: dict = field(default_factory=dict, init=False, repr=False)
    pooled_quantiles_: dict = field(default_factory=dict, init=False, repr=False)
    n_residuals_by_tenor_: dict = field(default_factory=dict, init=False, repr=False)
    n_pooled_residuals_: int = field(default=0, init=False, repr=False)

    def fit(
        self,
        train_pool: pd.DataFrame,
        *,
        point_model_factory: Callable[[], object],
        point_predict_fn: Callable[[object, pd.DataFrame], pd.Series],
    ) -> ResidualQuantileCalibrator:
        def _fit_fn(inner_train: pd.DataFrame):
            return point_model_factory().fit(inner_train)

        def _transform_fn(model, inner_test: pd.DataFrame) -> pd.DataFrame:
            out = inner_test.copy()
            preds = point_predict_fn(model, inner_test)
            out[RESIDUAL_COL] = out[self.target_col].to_numpy() - np.asarray(preds)
            return out

        residuals = walk_forward_transform(train_pool, fit_fn=_fit_fn, transform_fn=_transform_fn)
        if RESIDUAL_COL in residuals.columns:
            residuals = residuals.dropna(subset=[RESIDUAL_COL])
        else:
            # `walk_forward_transform` returned its "no strictly-prior
            # inner fold exists anywhere" empty frame (e.g. a training
            # pool spanning fewer than 2 distinct years) -- `transform_fn`
            # was never called, so `RESIDUAL_COL` was never added.
            residuals = residuals.iloc[0:0]

        self.n_pooled_residuals_ = len(residuals)
        if self.n_pooled_residuals_ == 0:
            # No strictly-prior inner fold existed anywhere in this
            # training pool (e.g. a training pool spanning fewer than
            # 2 distinct years) -- every quantile offset is 0.0 rather
            # than undefined, a documented, conservative degenerate
            # case that only affects the very earliest possible folds.
            self.pooled_quantiles_ = {level: 0.0 for level in self.quantile_levels}
        else:
            self.pooled_quantiles_ = {
                level: float(residuals[RESIDUAL_COL].quantile(level)) for level in self.quantile_levels
            }

        for tenor, group in residuals.groupby(TENOR_COL):
            n = len(group)
            self.n_residuals_by_tenor_[tenor] = n
            if n >= self.min_residuals_per_tenor:
                self.tenor_quantiles_[tenor] = {
                    level: float(group[RESIDUAL_COL].quantile(level)) for level in self.quantile_levels
                }

        self.is_fitted_ = True
        return self

    def predict_quantiles(self, test_df: pd.DataFrame, point_forecast: pd.Series) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("ResidualQuantileCalibrator.predict_quantiles called before fit")

        point = np.asarray(point_forecast)
        raw = np.empty((len(test_df), len(self.quantile_levels)))
        tenors = test_df[TENOR_COL].to_numpy()
        for j, level in enumerate(self.quantile_levels):
            offsets = np.array(
                [
                    self.tenor_quantiles_.get(t, {}).get(level, self.pooled_quantiles_[level])
                    for t in tenors
                ]
            )
            raw[:, j] = point + offsets

        sorted_matrix = np.sort(raw, axis=1)  # quantile-crossing fix, same convention as the GBM quantile model
        columns = {quantile_column_name(level): sorted_matrix[:, j] for j, level in enumerate(self.quantile_levels)}
        return pd.DataFrame(columns, index=test_df.index)
