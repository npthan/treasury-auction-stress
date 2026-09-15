"""Phase 7, Step 3: the Dealer Absorption Surprise / stress-event
CORRECTNESS GATE (`configs/phase_7_protocol.yml`'s `stress_event_gate`).

This module implements, in order:

1. A SAFE regime feature (`add_safe_regime_feature`), replacing Phase
   6's flagged-leaky `dealer_absorption.add_regime_feature` entirely --
   built from `treasury_auction_stress.features.safe_as_of_lookback.
   safe_as_of_trailing_mean`, safe by construction for every row
   regardless of same-day/closely-spaced auctions.
2. Cross-fitted (never in-sample) training-label construction
   (`build_cross_fitted_training_labels`), using
   `treasury_auction_stress.evaluation.walk_forward` so the threshold
   is never calibrated against a model's own in-sample residuals.
3. A frozen-per-fold test-label construction
   (`build_test_labels`) using an ordinary out-of-sample transform.
4. A data-sufficiency GATE (`evaluate_gate_for_fold` /
   `GateDecision`) that must pass BEFORE any classifier score is
   published for a given cutoff view -- per the task specification,
   failing this gate means the classifier is deferred, documented,
   never silently scored anyway.

Nothing here fits a model on the outer fold's own test year at all --
this module only ever produces LABELS (for both train and test rows);
`treasury_auction_stress.models.logistic_classifier` is the actual
classifier, fit separately once labels exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from treasury_auction_stress.evaluation.protocol7 import load_protocol7
from treasury_auction_stress.evaluation.timing import (
    RESULT_SAFE_AVAILABLE_DATE_COL,
    build_inner_holdout,
)
from treasury_auction_stress.evaluation.walk_forward import walk_forward_transform
from treasury_auction_stress.features.dealer_absorption import (
    DealerAbsorptionSurpriseModel,
    StressThreshold,
)
from treasury_auction_stress.features.safe_as_of_lookback import (
    safe_as_of_trailing_mean,
)

TENOR_COL = "tenor"
AUCTION_KEY_COL = "auction_key"
DATE_COL = "auction_date"
# Acceptance-review fix: genuinely sourced from the frozen protocol
# (configs/phase_7_protocol.yml's targets.primary), not a second,
# independently-hardcoded literal -- see run_evaluation_phase7.py's
# identical fix and tests/test_phase7_protocol_contract.py.
TARGET_COL = load_protocol7().primary_target

REGIME_WINDOW = 8
REGIME_SAFE_COL = "dealer_share_regime_trailing_mean_safe"

STRESS_PERCENTILE = 0.90

# Data-sufficiency criteria, pre-declared -- see
# configs/phase_7_protocol.yml's stress_event_gate section. Never
# adjusted after seeing a real positive count.
MIN_POOLED_POSITIVE_COUNT = 30
MIN_YEAR_POSITIVE_COUNT = 5


def add_safe_regime_feature(pool: pd.DataFrame, *, cutoff_col: str, window: int = REGIME_WINDOW) -> pd.DataFrame:
    """Replaces `dealer_absorption.add_regime_feature` entirely for
    Phase 7 purposes. Safe by construction for every row (see
    `safe_as_of_lookback`'s module docstring) -- computed once over
    `pool` (typically the full eligible sample for one cutoff view),
    never fold-scoped, because its safety does not depend on which
    other rows are present.
    """
    result = safe_as_of_trailing_mean(
        pool,
        group_col=TENOR_COL,
        value_col=TARGET_COL,
        key_col=AUCTION_KEY_COL,
        cutoff_col=cutoff_col,
        safe_date_col=RESULT_SAFE_AVAILABLE_DATE_COL,
        date_col=DATE_COL,
        n_lookback=window,
    )
    return result.rename(
        columns={
            f"{TARGET_COL}_safe_trailing_mean": REGIME_SAFE_COL,
            f"{TARGET_COL}_safe_trailing_n_used": f"{REGIME_SAFE_COL}_n_used",
        }
    )


def build_cross_fitted_training_labels(
    train_pool: pd.DataFrame, *, percentile: float = STRESS_PERCENTILE
) -> tuple[pd.Series, StressThreshold | None, pd.Series]:
    """Walk forward within `train_pool` (an OUTER fold's own training
    pool, never the outer test year) to build an out-of-fold Dealer
    Absorption Surprise series, fit `StressThreshold` on THAT (never
    on an in-sample surprise), and return:

    - `label`: a nullable-boolean `pd.Series` indexed like
      `train_pool`, `pd.NA` for any row with no cross-fitted surprise
      (the training pool's own earliest calendar year, or a row
      missing `offering_amt`) -- NEVER coerced to `False`.
    - `threshold`: the fitted `StressThreshold` (or `None` if zero
      cross-fitted surprise values exist anywhere -- an unusable-gate
      condition the caller must check).
    - `surprise`: the raw cross-fitted surprise series (same index
      convention as `label`), for descriptive reporting.
    """

    def _fit_fn(inner_train: pd.DataFrame) -> DealerAbsorptionSurpriseModel:
        return DealerAbsorptionSurpriseModel(regime_col=REGIME_SAFE_COL).fit(inner_train)

    def _transform_fn(model: DealerAbsorptionSurpriseModel, inner_test: pd.DataFrame) -> pd.DataFrame:
        return model.transform(inner_test)

    cross_fitted = walk_forward_transform(train_pool, fit_fn=_fit_fn, transform_fn=_transform_fn)

    surprise_full = pd.Series(pd.NA, index=train_pool.index, dtype="object")
    if "dealer_absorption_surprise" in cross_fitted.columns:
        surprise_full.loc[cross_fitted.index] = cross_fitted["dealer_absorption_surprise"]

    numeric_surprise = pd.to_numeric(surprise_full, errors="coerce")
    train_only = numeric_surprise.dropna()
    if train_only.empty:
        return pd.Series(pd.NA, index=train_pool.index, dtype="object"), None, surprise_full

    threshold = StressThreshold(percentile=percentile).fit(train_only)
    label = threshold.transform(numeric_surprise)
    return label, threshold, surprise_full


def build_test_labels(
    train_pool: pd.DataFrame, test_pool: pd.DataFrame, *, threshold: StressThreshold
) -> tuple[pd.Series, pd.Series]:
    """A SEPARATE `DealerAbsorptionSurpriseModel` fit on the ENTIRE
    outer training pool (legitimate: the test year is temporally
    entirely after it), transformed onto `test_pool`, labeled with the
    SAME cross-fitted `threshold` passed in (never re-fit on test
    outcomes). Returns `(label, surprise)`.
    """
    final_model = DealerAbsorptionSurpriseModel(regime_col=REGIME_SAFE_COL).fit(train_pool)
    transformed = final_model.transform(test_pool)
    surprise = transformed["dealer_absorption_surprise"]
    label = threshold.transform(surprise)
    return label, surprise


def build_inner_cv_safe_labels(outer_train_pool: pd.DataFrame, *, percentile: float = STRESS_PERCENTILE) -> pd.Series | None:
    """Acceptance-review addition, for the classifier's INNER
    hyperparameter selection ONLY -- never used for the final classifier
    fit or for outer test-year labels, both of which keep using
    `build_cross_fitted_training_labels`'s ordinary (outer-pool)
    threshold exactly as before.

    `build_cross_fitted_training_labels`'s own `StressThreshold` is fit
    on cross-fitted surprises pooled across the ENTIRE outer training
    pool -- including whatever year `evaluation.timing.build_inner_holdout`
    will later carve out as the inner-validation year. Scoring inner-
    validation rows with a label derived from a threshold that already
    "saw" their own year's surprise distribution is a nesting violation:
    verified directly against this project's real data (see
    the Phase 7 acceptance review) to shift the threshold by up
    to ~15% (relative) in early folds and flip at least one inner-
    validation row's label, before this fix.

    This function re-derives an INNER-consistent label by applying the
    exact same outer-level recipe one level down: split
    `outer_train_pool` via `build_inner_holdout` (the SAME split the
    classifier's own hyperparameter search uses), fit
    `build_cross_fitted_training_labels` on `inner_train` ALONE (so its
    threshold never sees `inner_val`'s own surprise), then label
    `inner_val` via `build_test_labels` using that inner-only threshold
    -- an ordinary out-of-sample transform, exactly mirroring the outer
    level's own `train`-vs-`test` split one level down.

    Returns `None` if the inner holdout itself is unusable (too few
    years/rows) or if `inner_train`'s own cross-fitted surprise is
    entirely empty -- callers must fall back to the ordinary
    (outer-threshold) label in that case, matching every other inner-CV
    fallback already established in this project
    (`models.logistic_classifier.FALLBACK_C`, etc.).
    """
    holdout = build_inner_holdout(outer_train_pool)
    if not holdout.usable:
        return None

    inner_train_label, inner_threshold, _inner_train_surprise = build_cross_fitted_training_labels(
        holdout.inner_train, percentile=percentile
    )
    if inner_threshold is None:
        return None

    inner_val_label, _inner_val_surprise = build_test_labels(holdout.inner_train, holdout.inner_val, threshold=inner_threshold)

    combined = pd.Series(pd.NA, index=outer_train_pool.index, dtype="object")
    combined.loc[inner_train_label.index] = inner_train_label
    combined.loc[inner_val_label.index] = inner_val_label
    return combined


@dataclass
class GateDecision:
    passes: bool
    reasons: list[str] = field(default_factory=list)
    pooled_positive_count: int = 0
    pooled_labeled_count: int = 0
    n_missing_training_labels: int = 0
    per_year_positive_counts: dict = field(default_factory=dict)
    per_year_labeled_counts: dict = field(default_factory=dict)
    years_with_reportable_metrics: list = field(default_factory=list)
    years_suppressed: list = field(default_factory=list)


def evaluate_gate(
    *,
    train_labels_by_fold: dict[int, pd.Series],
    test_labels_by_fold: dict[int, pd.Series],
    n_missing_training_labels_by_fold: dict[int, int],
) -> GateDecision:
    """Apply the pre-declared data-sufficiency criteria
    (`configs/phase_7_protocol.yml`'s
    `stress_event_gate.data_sufficiency_criteria_checked_before_any_classifier_score_is_published`)
    across every complete-year fold's own TEST labels (the classifier's
    actual evaluated outcomes) for one cutoff view. `train_labels_by_fold`
    is accepted for the missing-label-count check only.
    """
    reasons: list[str] = []
    per_year_positive = {}
    per_year_labeled = {}
    years_reportable = []
    years_suppressed = []

    pooled_positive = 0
    pooled_labeled = 0
    for year, labels in test_labels_by_fold.items():
        numeric = pd.to_numeric(labels, errors="coerce")
        labeled = numeric.dropna()
        n_labeled = len(labeled)
        n_positive = int(labeled.sum())
        per_year_positive[year] = n_positive
        per_year_labeled[year] = n_labeled
        pooled_positive += n_positive
        pooled_labeled += n_labeled
        if n_positive >= MIN_YEAR_POSITIVE_COUNT:
            years_reportable.append(year)
        else:
            years_suppressed.append(year)

    n_missing_training_labels = int(sum(n_missing_training_labels_by_fold.values()))

    passes = pooled_positive >= MIN_POOLED_POSITIVE_COUNT
    if not passes:
        reasons.append(
            f"pooled positive-label count across all complete test years is {pooled_positive}, "
            f"below the required minimum of {MIN_POOLED_POSITIVE_COUNT} -- classifier scores are not published"
        )
    if years_suppressed:
        reasons.append(
            f"years with fewer than {MIN_YEAR_POSITIVE_COUNT} positive labels have their own per-year "
            f"metrics suppressed: {years_suppressed}"
        )

    return GateDecision(
        passes=passes,
        reasons=reasons,
        pooled_positive_count=pooled_positive,
        pooled_labeled_count=pooled_labeled,
        n_missing_training_labels=n_missing_training_labels,
        per_year_positive_counts=per_year_positive,
        per_year_labeled_counts=per_year_labeled,
        years_with_reportable_metrics=years_reportable,
        years_suppressed=years_suppressed,
    )


def compute_classifier_metrics(
    df: pd.DataFrame,
    *,
    proba_col: str = "predicted_proba",
    label_col: str = "actual_label",
    no_skill_proba_col: str | None = "train_prevalence",
) -> dict:
    """Prevalence, Brier score, PR-AUC (with the positive count that
    made it computable), and a confusion table at each row's OWN
    `decision_threshold`-derived `predicted_label` column. Returns
    `n=0`/`None` metrics for an empty or all-one-class `df` rather than
    letting `average_precision_score` raise -- PR-AUC is undefined with
    a single class present, and this is reported as `None`, not a
    fabricated number.

    `no_skill_proba_col`, when present in `df` (the default,
    `train_prevalence`), also reports the Brier score of a trivial
    "no-skill" forecast that always predicts that row's own fold's
    TRAINING-label prevalence (never 0.5, never the pooled/global
    prevalence) -- the honest baseline a classifier must beat to be
    worth reporting as having any calibrated skill at all. A
    classifier's own Brier score ABOVE (worse than) this baseline means
    it is poorly calibrated, even if its PR-AUC ranking is non-trivial
    -- both numbers are reported together so neither is read alone.
    """
    clean = df.dropna(subset=[label_col])
    if clean.empty:
        return {
            "n": 0,
            "n_positive": 0,
            "prevalence": None,
            "brier_score": None,
            "pr_auc": None,
            "no_skill_brier_score": None,
        }

    labels = clean[label_col].astype(int)
    proba = clean[proba_col].astype(float)
    n_positive = int(labels.sum())

    metrics: dict = {
        "n": len(clean),
        "n_positive": n_positive,
        "prevalence": float(labels.mean()),
        "brier_score": float(brier_score_loss(labels, proba)),
        "pr_auc": float(average_precision_score(labels, proba)) if labels.nunique() >= 2 else None,
    }
    if no_skill_proba_col is not None and no_skill_proba_col in clean.columns:
        no_skill_proba = clean[no_skill_proba_col].astype(float)
        metrics["no_skill_brier_score"] = float(brier_score_loss(labels, no_skill_proba))
        metrics["worse_than_no_skill"] = metrics["brier_score"] > metrics["no_skill_brier_score"]
    else:
        metrics["no_skill_brier_score"] = None

    if "predicted_label" in clean.columns:
        predicted = clean["predicted_label"].astype(bool)
        actual = labels.astype(bool)
        metrics["confusion_true_positive"] = int((predicted & actual).sum())
        metrics["confusion_false_positive"] = int((predicted & ~actual).sum())
        metrics["confusion_true_negative"] = int((~predicted & ~actual).sum())
        metrics["confusion_false_negative"] = int((~predicted & actual).sum())

    return metrics


def compute_calibration_table(
    df: pd.DataFrame, *, proba_col: str = "predicted_proba", label_col: str = "actual_label", n_bins: int = 5
) -> pd.DataFrame:
    """Predicted-probability decile (or `n_bins`-ile) vs. observed
    frequency -- a reliability table, computed only on rows with a
    non-missing label. Bin edges are quantiles of the predicted
    probability itself (equal-COUNT bins), never equal-width bins that
    could leave a bin empty given how few positive labels this project
    has per year.
    """
    clean = df.dropna(subset=[label_col]).copy()
    if len(clean) < n_bins:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted_proba", "observed_frequency"])

    clean["_bin"] = pd.qcut(clean[proba_col], q=n_bins, duplicates="drop")
    rows = []
    for bin_label, group in clean.groupby("_bin", observed=True):
        rows.append(
            {
                "bin": str(bin_label),
                "n": len(group),
                "mean_predicted_proba": float(group[proba_col].mean()),
                "observed_frequency": float(group[label_col].astype(int).mean()),
            }
        )
    return pd.DataFrame(rows)
