"""Phase 7: the single, pre-declared nonlinear point-forecast challenger
(`challenger_gbm_core`, `configs/phase_7_protocol.yml`'s
`gbm_challenger`) and its quantile-regression sibling
(`quantile_gbm_core`, `probabilistic.prespecified_challenger`).

`sklearn.ensemble.HistGradientBoostingRegressor` is used because it is
already a project dependency (no new package added), handles missing
values natively (no `SimpleImputer` needed -- a disclosed, deliberate
difference from `treasury_auction_stress.models.linear_models`'s
pipeline), and supports both `loss="squared_error"` (the point
challenger) and `loss="quantile"` (the probabilistic challenger) in one
estimator family.

All preprocessing and hyperparameter selection happen strictly inside
one fold's own training pool, mirroring `linear_models.py`'s discipline
exactly: the 100%-missing-in-training-fold column-drop rule is applied
first, then the same nested inner-holdout hyperparameter search
(`treasury_auction_stress.evaluation.timing.build_inner_holdout`) is
used, scored by MAE on the target's own raw units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from treasury_auction_stress.evaluation.probabilistic_metrics import (
    quantile_column_name,
)
from treasury_auction_stress.evaluation.timing import build_inner_holdout
from treasury_auction_stress.models.linear_models import (
    CATEGORICAL_COLUMNS,
    drop_all_missing_training_columns,
)

FALLBACK_GBM_HYPERPARAMS: dict = {
    "max_iter": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}

GBM_HYPERPARAM_GRID: tuple[dict, ...] = tuple(
    {
        "max_iter": max_iter,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": 42,
    }
    for max_iter in (50, 100)
    for max_depth in (2, 3)
    for learning_rate in (0.05, 0.1)
)


def _prepare_categorical_frame(
    df: pd.DataFrame, feature_cols: list[str], *, train_categories: dict[str, list] | None = None
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Cast every column in `feature_cols` that is one of
    `linear_models.CATEGORICAL_COLUMNS` to a pandas `category` dtype so
    `HistGradientBoostingRegressor(categorical_features="from_dtype")`
    treats it natively (no one-hot expansion). When `train_categories`
    is given (the test-side call), the SAME category set learned from
    training is applied -- a category value never seen in training
    becomes `NaN` (routed by HGBR's native missing-value handling,
    never raising and never silently inventing a new category).
    """
    out = df[feature_cols].copy()
    categories: dict[str, list] = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in out.columns:
            continue
        if train_categories is not None:
            cats = train_categories[col]
            # A category value never seen in training is set to NaN
            # BEFORE constructing the Categorical (rather than passed
            # through and relying on `categories=` to silently drop
            # it) -- explicit, and avoids pandas' own deprecation
            # warning for constructing a Categorical with out-of-
            # vocabulary values.
            out[col] = out[col].where(out[col].isin(cats) | out[col].isna())
        else:
            cats = sorted(out[col].dropna().unique().tolist())
            categories[col] = cats
        out[col] = pd.Categorical(out[col], categories=cats)
    return out, categories


def _fit_one_gbm(
    train_X: pd.DataFrame, train_y: pd.Series, hyperparams: dict, *, quantile: float | None
) -> HistGradientBoostingRegressor:
    params = dict(hyperparams)
    if quantile is not None:
        params["loss"] = "quantile"
        params["quantile"] = quantile
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", **params)
    model.fit(train_X, train_y)
    return model


def select_gbm_hyperparameters_via_inner_cv(
    train_df: pd.DataFrame, *, feature_cols: list[str], target_col: str
) -> tuple[dict, bool, str | None]:
    """Identical mechanism to
    `linear_models.select_hyperparameters_via_inner_cv`, adapted for
    the GBM grid. Never looks at the outer fold's own test set.
    """
    holdout = build_inner_holdout(train_df)
    if not holdout.usable:
        return dict(FALLBACK_GBM_HYPERPARAMS), False, holdout.reason_if_unusable

    inner_train = holdout.inner_train.dropna(subset=[target_col])
    inner_val = holdout.inner_val.dropna(subset=[target_col])
    if inner_train.empty or inner_val.empty:
        return dict(FALLBACK_GBM_HYPERPARAMS), False, "inner train or inner validation set has no non-missing target values"

    kept_cols, _ = drop_all_missing_training_columns(inner_train, feature_cols)
    if not kept_cols:
        return dict(FALLBACK_GBM_HYPERPARAMS), False, "no usable predictor columns in inner training pool"

    inner_train_X, train_categories = _prepare_categorical_frame(inner_train, kept_cols)
    inner_val_X, _ = _prepare_categorical_frame(inner_val, kept_cols, train_categories=train_categories)

    best_config, best_score = None, np.inf
    for candidate in GBM_HYPERPARAM_GRID:
        model = _fit_one_gbm(inner_train_X, inner_train[target_col].astype("float64"), candidate, quantile=None)
        preds = model.predict(inner_val_X)
        score = mean_absolute_error(inner_val[target_col].astype("float64"), preds)
        if score < best_score:
            best_score, best_config = score, candidate

    return best_config, True, None


@dataclass
class GBMFitResult:
    predictions: pd.Series
    dropped_all_missing_columns: list[str] = field(default_factory=list)
    kept_feature_columns: list[str] = field(default_factory=list)
    selected_hyperparams: dict = field(default_factory=dict)
    used_inner_cv: bool = False
    inner_cv_reason_if_not_used: str | None = None
    n_train_rows: int = 0


def fit_predict_gbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    hyperparams_override: dict | None = None,
) -> GBMFitResult:
    """Fit `challenger_gbm_core` (point forecast, `loss="squared_error"`)
    on `train_df` and predict `test_df`. When `hyperparams_override` is
    given (used by the quantile sibling below, which reuses the point
    model's own selected tree hyperparameters rather than running a
    second search), hyperparameter selection is skipped entirely.
    """
    train_clean = train_df.dropna(subset=[target_col])
    if train_clean.empty:
        raise ValueError("fit_predict_gbm: no non-missing training targets")

    kept_cols, dropped_cols = drop_all_missing_training_columns(train_clean, feature_cols)
    if not kept_cols:
        raise ValueError("fit_predict_gbm: every requested feature column is 100% missing in training data")

    if hyperparams_override is not None:
        selected_hyperparams, used_inner_cv, reason = dict(hyperparams_override), False, (
            "hyperparameters reused from challenger_gbm_core's own inner-CV selection for this fold"
        )
    else:
        selected_hyperparams, used_inner_cv, reason = select_gbm_hyperparameters_via_inner_cv(
            train_clean, feature_cols=kept_cols, target_col=target_col
        )

    train_X, train_categories = _prepare_categorical_frame(train_clean, kept_cols)
    test_X, _ = _prepare_categorical_frame(test_df, kept_cols, train_categories=train_categories)

    model = _fit_one_gbm(train_X, train_clean[target_col].astype("float64"), selected_hyperparams, quantile=None)
    predictions = pd.Series(model.predict(test_X), index=test_df.index)

    return GBMFitResult(
        predictions=predictions,
        dropped_all_missing_columns=dropped_cols,
        kept_feature_columns=kept_cols,
        selected_hyperparams=selected_hyperparams,
        used_inner_cv=used_inner_cv,
        inner_cv_reason_if_not_used=reason,
        n_train_rows=len(train_clean),
    )


def fit_predict_gbm_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    quantile_levels: tuple[float, ...],
    tree_hyperparams: dict,
) -> pd.DataFrame:
    """`quantile_gbm_core`: one `HistGradientBoostingRegressor(loss=
    "quantile")` fit per level in `quantile_levels`, all reusing the
    SAME `tree_hyperparams` (the point challenger's own inner-CV
    selection for this fold) -- no separate hyperparameter search per
    quantile level, per `configs/phase_7_protocol.yml`.

    Returns a DataFrame indexed like `test_df`, one column per quantile
    level (named via `evaluation.probabilistic_metrics.
    quantile_column_name`), with predicted quantiles SORTED row-wise
    (standard, disclosed fix for quantile crossing across
    independently-fit quantile regressions -- never re-estimated).
    """
    train_clean = train_df.dropna(subset=[target_col])
    if train_clean.empty:
        raise ValueError("fit_predict_gbm_quantiles: no non-missing training targets")

    kept_cols, _ = drop_all_missing_training_columns(train_clean, feature_cols)
    if not kept_cols:
        raise ValueError("fit_predict_gbm_quantiles: every requested feature column is 100% missing in training data")

    train_X, train_categories = _prepare_categorical_frame(train_clean, kept_cols)
    test_X, _ = _prepare_categorical_frame(test_df, kept_cols, train_categories=train_categories)
    train_y = train_clean[target_col].astype("float64")

    raw = {}
    for level in quantile_levels:
        model = _fit_one_gbm(train_X, train_y, tree_hyperparams, quantile=level)
        raw[level] = model.predict(test_X)

    raw_matrix = np.column_stack([raw[level] for level in quantile_levels])
    sorted_matrix = np.sort(raw_matrix, axis=1)
    columns = {quantile_column_name(level): sorted_matrix[:, i] for i, level in enumerate(quantile_levels)}
    return pd.DataFrame(columns, index=test_df.index)
