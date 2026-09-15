"""Phase 6: Ridge / Elastic Net linear models (models 5-7 of
`configs/phase_6_evaluation.yml`), with every learned preprocessing
step (imputation, scaling, categorical encoding, missingness
indicators, hyperparameter selection) fit strictly inside one fold's
own training pool -- never on the full Phase 5 matrix.

`fit_predict_linear` is the single entry point the evaluation driver
calls: it drops any predictor 100% missing in the training pool
(pre-declared rule, `configs/phase_6_evaluation.yml`'s
`missing_predictor_handling`), builds a fresh scikit-learn pipeline,
runs the nested inner-holdout hyperparameter search (or the
pre-declared fixed fallback when there is not enough history), refits
on the full training pool with the selected hyperparameters, and
predicts on the test pool. Every one of those steps is redone from
scratch for every fold -- nothing here is ever fit once and reused
across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from treasury_auction_stress.evaluation.timing import build_inner_holdout

# `direct_contract_mapping_status` (extended tier) is a string-valued
# CFTC audit label (e.g. "direct_contract_available"), not numeric --
# see configs/phase_5_features.yml. Every other core/extended predictor
# is numeric or boolean.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("tenor", "direct_contract_mapping_status")

FALLBACK_RIDGE_ALPHA = 1.0
FALLBACK_ELASTIC_NET_ALPHA = 0.1
FALLBACK_ELASTIC_NET_L1_RATIO = 0.5

RIDGE_ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
ELASTIC_NET_ALPHA_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
ELASTIC_NET_L1_RATIO_GRID: tuple[float, ...] = (0.1, 0.5, 0.9)


def drop_all_missing_training_columns(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Pre-declared rule: a predictor 100% missing in `train_df` is
    dropped -- returns `(kept_cols, dropped_cols)`. `feature_cols`
    order is preserved in `kept_cols`.
    """
    dropped = [c for c in feature_cols if train_df[c].isna().all()]
    kept = [c for c in feature_cols if c not in dropped]
    return kept, dropped


def _make_estimator(estimator_type: str, hyperparams: dict):
    if estimator_type == "ridge":
        return Ridge(alpha=hyperparams["alpha"])
    if estimator_type == "elastic_net":
        return ElasticNet(alpha=hyperparams["alpha"], l1_ratio=hyperparams["l1_ratio"], max_iter=10_000)
    raise ValueError(f"unknown estimator_type {estimator_type!r}")


def build_pipeline(feature_cols: list[str], estimator_type: str, hyperparams: dict) -> Pipeline:
    """A fresh (unfitted) scikit-learn pipeline: numeric columns get
    training-fold median imputation (plus a missingness indicator) and
    standard scaling; `tenor` (if present among `feature_cols`) gets
    one-hot encoding with `handle_unknown="ignore"` so a category never
    seen in this fold's training data (e.g. a tenor's first-ever
    appearance) is encoded as all-zero at test time rather than raising
    or requiring a future category vocabulary. `is_reopening` (already
    boolean/{0,1}) passes through the numeric branch unchanged.
    """
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))

    preprocess = ColumnTransformer(transformers)
    estimator = _make_estimator(estimator_type, hyperparams)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _hyperparameter_grid(estimator_type: str) -> list[dict]:
    if estimator_type == "ridge":
        return [{"alpha": a} for a in RIDGE_ALPHA_GRID]
    if estimator_type == "elastic_net":
        return [
            {"alpha": a, "l1_ratio": l1}
            for a, l1 in product(ELASTIC_NET_ALPHA_GRID, ELASTIC_NET_L1_RATIO_GRID)
        ]
    raise ValueError(f"unknown estimator_type {estimator_type!r}")


def _fallback_hyperparams(estimator_type: str) -> dict:
    if estimator_type == "ridge":
        return {"alpha": FALLBACK_RIDGE_ALPHA}
    if estimator_type == "elastic_net":
        return {"alpha": FALLBACK_ELASTIC_NET_ALPHA, "l1_ratio": FALLBACK_ELASTIC_NET_L1_RATIO}
    raise ValueError(f"unknown estimator_type {estimator_type!r}")


@dataclass
class LinearFitResult:
    predictions: pd.Series  # unclipped, indexed like test_df
    dropped_all_missing_columns: list[str] = field(default_factory=list)
    kept_feature_columns: list[str] = field(default_factory=list)
    selected_hyperparams: dict = field(default_factory=dict)
    used_inner_cv: bool = False
    inner_cv_reason_if_not_used: str | None = None
    n_train_rows: int = 0


def select_hyperparameters_via_inner_cv(
    train_df: pd.DataFrame, *, feature_cols: list[str], target_col: str, estimator_type: str
) -> tuple[dict, bool, str | None]:
    """Nested inner-holdout hyperparameter selection (see
    `configs/phase_6_evaluation.yml`'s `hyperparameter_selection`).
    Returns `(selected_hyperparams, used_inner_cv, reason_if_not_used)`.
    Never looks at the outer fold's own test set -- only `train_df`
    (the outer training pool) is used here.
    """
    holdout = build_inner_holdout(train_df)
    if not holdout.usable:
        return _fallback_hyperparams(estimator_type), False, holdout.reason_if_unusable

    inner_train = holdout.inner_train.dropna(subset=[target_col])
    inner_val = holdout.inner_val.dropna(subset=[target_col])
    if inner_train.empty or inner_val.empty:
        return (
            _fallback_hyperparams(estimator_type),
            False,
            "inner train or inner validation set has no non-missing target values",
        )

    kept_cols, _ = drop_all_missing_training_columns(inner_train, feature_cols)
    if not kept_cols:
        return _fallback_hyperparams(estimator_type), False, "no usable predictor columns in inner training pool"

    best_config, best_score = None, np.inf
    for candidate in _hyperparameter_grid(estimator_type):
        pipeline = build_pipeline(kept_cols, estimator_type, candidate)
        pipeline.fit(inner_train[kept_cols], inner_train[target_col].astype("float64"))
        preds = pipeline.predict(inner_val[kept_cols])
        score = mean_absolute_error(inner_val[target_col].astype("float64"), preds)
        if score < best_score:
            best_score, best_config = score, candidate

    return best_config, True, None


def fit_predict_linear(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    estimator_type: str,
) -> LinearFitResult:
    """Fit one linear model on `train_df` and predict `test_df`. All
    preprocessing and hyperparameter selection happens strictly inside
    this call, on `train_df` only.
    """
    train_clean = train_df.dropna(subset=[target_col])
    if train_clean.empty:
        raise ValueError("fit_predict_linear: no non-missing training targets")

    kept_cols, dropped_cols = drop_all_missing_training_columns(train_clean, feature_cols)
    if not kept_cols:
        raise ValueError("fit_predict_linear: every requested feature column is 100% missing in training data")

    selected_hyperparams, used_inner_cv, reason = select_hyperparameters_via_inner_cv(
        train_clean, feature_cols=kept_cols, target_col=target_col, estimator_type=estimator_type
    )

    pipeline = build_pipeline(kept_cols, estimator_type, selected_hyperparams)
    pipeline.fit(train_clean[kept_cols], train_clean[target_col].astype("float64"))
    predictions = pd.Series(pipeline.predict(test_df[kept_cols]), index=test_df.index)

    return LinearFitResult(
        predictions=predictions,
        dropped_all_missing_columns=dropped_cols,
        kept_feature_columns=kept_cols,
        selected_hyperparams=selected_hyperparams,
        used_inner_cv=used_inner_cv,
        inner_cv_reason_if_not_used=reason,
        n_train_rows=len(train_clean),
    )
