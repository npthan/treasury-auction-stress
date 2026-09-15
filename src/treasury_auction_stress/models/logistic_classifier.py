"""Phase 7: `stress_classifier_logistic_core`
(`configs/phase_7_protocol.yml`'s `stress_event_gate.classifier`) --
the modest, pre-declared classification benchmark used ONLY if
`treasury_auction_stress.evaluation.stress_event`'s data-sufficiency
gate passes.

Mirrors `treasury_auction_stress.models.linear_models`'s preprocessing
and nested-inner-holdout hyperparameter-selection discipline exactly
(imputation + scaling for numeric columns, one-hot encoding for
categorical columns, all fit strictly inside the outer fold's own
training pool), swapping Ridge/Elastic Net's MAE selection metric for
Brier score (the standard proper scoring rule for a probabilistic
binary label) and Ridge/Elastic Net's regression estimator for
`sklearn.linear_model.LogisticRegression`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from treasury_auction_stress.evaluation.timing import build_inner_holdout
from treasury_auction_stress.models.linear_models import (
    CATEGORICAL_COLUMNS,
    drop_all_missing_training_columns,
)

FALLBACK_C = 1.0
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)


def build_classifier_pipeline(feature_cols: list[str], C: float) -> Pipeline:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))

    preprocess = ColumnTransformer(transformers)
    # `l1_ratio=0.0` is scikit-learn's current (>=1.8) spelling for pure
    # L2 regularization -- the `penalty="l2"` spelling is deprecated.
    estimator = LogisticRegression(C=C, l1_ratio=0.0, max_iter=2000)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def select_c_via_inner_cv(
    train_df: pd.DataFrame, *, feature_cols: list[str], label_col: str, inner_label_col: str | None = None
) -> tuple[float, bool, str | None]:
    """Nested inner-holdout selection of `C`, scored by Brier score
    (mean squared error between predicted probability and the {0,1}
    label) on the inner validation set. Never looks at the outer
    fold's own test set.

    `inner_label_col`, when given, is used INSTEAD of `label_col` for
    both fitting the inner-training model and scoring it on the inner-
    validation set. This matters for the stress-event classifier
    specifically: `label_col` (the outer training label) is built from
    a threshold fit on cross-fitted surprises pooled across the WHOLE
    outer training pool -- including whatever year this very function's
    own `build_inner_holdout` call will carve out as the inner-
    validation year. Scoring inner-validation rows with a label derived
    from a threshold that already "saw" their own year's surprise
    distribution is a nesting violation (acceptance-review finding,
    verified to shift the threshold by up to ~15% in early folds and
    flip real labels -- see the Phase 7 acceptance review and
    `evaluation.stress_event.build_inner_cv_safe_labels`, which supplies
    `inner_label_col` for exactly this reason). Ordinary (non-classifier)
    callers omit `inner_label_col` and get the prior behavior exactly.
    """
    holdout = build_inner_holdout(train_df)
    if not holdout.usable:
        return FALLBACK_C, False, holdout.reason_if_unusable

    scoring_col = inner_label_col if inner_label_col is not None else label_col
    inner_train = holdout.inner_train.dropna(subset=[scoring_col])
    inner_val = holdout.inner_val.dropna(subset=[scoring_col])
    if inner_train.empty or inner_val.empty:
        return FALLBACK_C, False, "inner train or inner validation set has no non-missing label values"
    if inner_train[scoring_col].nunique() < 2:
        return FALLBACK_C, False, "inner training pool's label has fewer than 2 distinct classes"
    if inner_val[scoring_col].nunique() < 2:
        return FALLBACK_C, False, "inner validation pool's label has fewer than 2 distinct classes"

    kept_cols, _ = drop_all_missing_training_columns(inner_train, feature_cols)
    if not kept_cols:
        return FALLBACK_C, False, "no usable predictor columns in inner training pool"

    best_c, best_score = None, np.inf
    for candidate_c in C_GRID:
        pipeline = build_classifier_pipeline(kept_cols, candidate_c)
        pipeline.fit(inner_train[kept_cols], inner_train[scoring_col].astype(int))
        proba = pipeline.predict_proba(inner_val[kept_cols])[:, 1]
        score = brier_score_loss(inner_val[scoring_col].astype(int), proba)
        if score < best_score:
            best_score, best_c = score, candidate_c

    return best_c, True, None


@dataclass
class ClassifierFitResult:
    predicted_proba: pd.Series
    train_prevalence: float
    decision_threshold: float
    predicted_label: pd.Series
    selected_c: float = FALLBACK_C
    used_inner_cv: bool = False
    inner_cv_reason_if_not_used: str | None = None
    dropped_all_missing_columns: list[str] = field(default_factory=list)
    kept_feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0


def fit_predict_logistic_classifier(
    train_df: pd.DataFrame, test_df: pd.DataFrame, *, feature_cols: list[str], label_col: str, inner_label_col: str | None = None
) -> ClassifierFitResult:
    """Fit on `train_df` rows with a non-missing `label_col` (rows
    without a cross-fitted label -- see
    `treasury_auction_stress.evaluation.stress_event` -- are excluded
    from fitting entirely, never coerced to the negative class) and
    predict every row of `test_df`. The decision rule for the
    confusion table is fixed and pre-declared: predicted probability
    `>=` this fold's OWN training-label prevalence -- never 0.5 by
    default, never chosen by inspecting the test year's own outcomes.

    `inner_label_col`, if given, is forwarded to `select_c_via_inner_cv`
    so hyperparameter SELECTION uses an inner-CV-safe label (see that
    function's docstring) while the FINAL classifier fit below still
    uses `label_col` -- the ordinary, non-leaking outer training label,
    unchanged.
    """
    train_clean = train_df.dropna(subset=[label_col])
    if train_clean.empty:
        raise ValueError("fit_predict_logistic_classifier: no rows with a non-missing label")
    if train_clean[label_col].nunique() < 2:
        raise ValueError("fit_predict_logistic_classifier: training label has fewer than 2 distinct classes")

    kept_cols, dropped_cols = drop_all_missing_training_columns(train_clean, feature_cols)
    if not kept_cols:
        raise ValueError("fit_predict_logistic_classifier: every requested feature column is 100% missing")

    selected_c, used_inner_cv, reason = select_c_via_inner_cv(
        train_clean, feature_cols=kept_cols, label_col=label_col, inner_label_col=inner_label_col
    )

    pipeline = build_classifier_pipeline(kept_cols, selected_c)
    pipeline.fit(train_clean[kept_cols], train_clean[label_col].astype(int))

    train_prevalence = float(train_clean[label_col].astype(int).mean())
    proba = pd.Series(pipeline.predict_proba(test_df[kept_cols])[:, 1], index=test_df.index)
    predicted_label = proba >= train_prevalence

    return ClassifierFitResult(
        predicted_proba=proba,
        train_prevalence=train_prevalence,
        decision_threshold=train_prevalence,
        predicted_label=predicted_label,
        selected_c=selected_c,
        used_inner_cv=used_inner_cv,
        inner_cv_reason_if_not_used=reason,
        dropped_all_missing_columns=dropped_cols,
        kept_feature_columns=kept_cols,
        n_train_rows=len(train_clean),
    )
