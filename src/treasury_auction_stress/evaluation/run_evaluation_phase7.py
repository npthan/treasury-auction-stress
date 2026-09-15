"""Phase 7: the evaluation driver. Ties together every Phase 7 module
(`models.adaptive_baselines`, `models.gbm`, `models.shrinkage`,
`models.quantile_residual`, `models.logistic_classifier`,
`evaluation.stress_event`) into three out-of-sample tables: point
forecasts, probabilistic (quantile) forecasts, and -- only if the
stress-event gate passes for a given cutoff view -- classifier scores.

Every point/probabilistic model is fit exactly once per (cutoff view,
fold, target) and used to predict every auction in that fold's test
year, mirroring Phase 6's `frozen_annual_models` discipline exactly.
`baseline_recent_history_adaptive` is the one documented exception:
its OWN prediction for a given test row can use an already-resolved
EARLIER test-year row's result, per `configs/phase_7_protocol.yml`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from treasury_auction_stress.evaluation.protocol import (
    load_protocol,
    resolve_tier_feature_columns,
)
from treasury_auction_stress.evaluation.protocol7 import Protocol7, load_protocol7
from treasury_auction_stress.evaluation.stress_event import (
    add_safe_regime_feature,
    build_cross_fitted_training_labels,
    build_inner_cv_safe_labels,
    build_test_labels,
    evaluate_gate,
)
from treasury_auction_stress.evaluation.timing import (
    Fold,
    add_result_safe_available_date,
    build_fold,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)
from treasury_auction_stress.features.feature_manifest import FeatureEntry
from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL
from treasury_auction_stress.models.adaptive_baselines import (
    AdaptiveRecentHistoryBaseline,
)
from treasury_auction_stress.models.baselines import RecentHistoryBaseline
from treasury_auction_stress.models.gbm import (
    fit_predict_gbm,
    fit_predict_gbm_quantiles,
)
from treasury_auction_stress.models.logistic_classifier import (
    fit_predict_logistic_classifier,
)
from treasury_auction_stress.models.quantile_residual import ResidualQuantileCalibrator
from treasury_auction_stress.models.shrinkage import ShrinkageTenorReopeningBaseline

CUTOFF_COL_BY_VIEW: dict[str, str] = {
    "announcement": ANNOUNCEMENT_CUTOFF_COL,
    "pre_auction": PRE_AUCTION_CUTOFF_COL,
}

# Acceptance-review fix: genuinely sourced from the frozen protocol
# (configs/phase_7_protocol.yml's targets.primary), not a second,
# independently-hardcoded literal that could silently drift from it --
# tests/test_phase7_protocol_contract.py asserts this equality directly
# as a second, redundant line of defense.
TARGET_COL = load_protocol7().primary_target
N_LOOKBACK = 8
GBM_FEATURE_TIER = "core"

POINT_MODEL_IDS: tuple[str, ...] = (
    "baseline_recent_history_frozen",
    "baseline_recent_history_adaptive",
    "challenger_gbm_core",
    "challenger_shrinkage_tenor_reopening",
)

CLIP_LOWER, CLIP_UPPER = 0.0, 1.0  # configs/phase_7_protocol.yml targets.clipping

POINT_PREDICTION_COLUMNS: tuple[str, ...] = (
    AUCTION_KEY_COL,
    "cutoff_view",
    "target_name",
    "model_id",
    "test_year",
    "fold_id",
    "is_provisional",
    "tenor",
    "is_reopening",
    "actual",
    "forecast",
    "forecast_unclipped",
    "was_clipped",
    "test_prediction_cutoff",
    "training_row_count",
    "model_config",
)

PROBABILISTIC_MODEL_IDS: tuple[str, ...] = ("quantile_recent_history_residual", "quantile_gbm_core")


def build_modeling_frame(matrix: pd.DataFrame, target_table: pd.DataFrame) -> pd.DataFrame:
    """Identical construction to
    `treasury_auction_stress.evaluation.run_evaluation.build_modeling_frame`
    -- reused, not re-derived, so Phase 7's frame is bit-for-bit the
    same as Phase 6's own for identical inputs."""
    merged = matrix.merge(target_table, on=AUCTION_KEY_COL, how="inner", validate="one_to_one")
    if len(merged) != len(matrix) or len(merged) != len(target_table):
        raise AssertionError("build_modeling_frame: matrix and target table key sets disagree")
    return add_result_safe_available_date(merged, date_col="auction_date")


def add_safe_regime_features_both_cutoffs(
    announcement_frame: pd.DataFrame, pre_auction_frame: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The safe regime feature is computed exactly once per cutoff view,
    over the FULL eligible sample (never fold-scoped, never recomputed
    on an already-processed frame -- see
    `safe_as_of_lookback.safe_as_of_trailing_mean`'s duplicate-column
    guard)."""
    return {
        "announcement": add_safe_regime_feature(announcement_frame, cutoff_col=ANNOUNCEMENT_CUTOFF_COL),
        "pre_auction": add_safe_regime_feature(pre_auction_frame, cutoff_col=PRE_AUCTION_CUTOFF_COL),
    }


def _all_fold_years(protocol7: Protocol7) -> list[tuple[int, bool]]:
    return [(y, False) for y in protocol7.complete_test_years] + [(protocol7.provisional_test_year, True)]


def _core_feature_columns(entries: list[FeatureEntry]) -> list[str]:
    protocol6 = load_protocol()
    return resolve_tier_feature_columns(protocol6, GBM_FEATURE_TIER, entries)


def _point_row(
    *, fold: Fold, cutoff_view: str, model_id: str, forecast: pd.Series, config: dict, is_provisional: bool
) -> pd.DataFrame:
    unclipped = pd.Series(forecast.to_numpy(), index=fold.test.index)
    clipped = unclipped.clip(lower=CLIP_LOWER, upper=CLIP_UPPER)
    was_clipped = (clipped != unclipped) & unclipped.notna()
    cutoff_col = CUTOFF_COL_BY_VIEW[cutoff_view]
    return pd.DataFrame(
        {
            AUCTION_KEY_COL: fold.test[AUCTION_KEY_COL].to_numpy(),
            "cutoff_view": cutoff_view,
            "target_name": TARGET_COL,
            "model_id": model_id,
            "test_year": fold.test_year,
            "fold_id": f"{fold.test_year}_provisional" if is_provisional else str(fold.test_year),
            "is_provisional": is_provisional,
            "tenor": fold.test["tenor"].to_numpy(),
            "is_reopening": fold.test["is_reopening"].to_numpy(),
            "actual": fold.test[TARGET_COL].to_numpy(),
            "forecast": clipped.to_numpy(),
            "forecast_unclipped": unclipped.to_numpy(),
            "was_clipped": was_clipped.to_numpy(),
            "test_prediction_cutoff": fold.test[cutoff_col].to_numpy(),
            "training_row_count": len(fold.train),
            "model_config": json.dumps(config, default=str),
        }
    )[list(POINT_PREDICTION_COLUMNS)]


def _prior_years_candidate_pool(frame: pd.DataFrame, *, test_year: int, date_col: str = "auction_date") -> pd.DataFrame:
    """Every row of `frame` with `date_col` in a year strictly before
    `test_year` -- deliberately NOT restricted by `fit_origin`/
    `result_safe_available_date`, unlike `fold.train`. Used only as the
    ADAPTIVE baseline's candidate-lookup pool (see
    `models.adaptive_baselines.AdaptiveRecentHistoryBaseline`'s
    acceptance-review docstring addition): a prior-year row excluded
    from `fold.train` because its own result was not yet safely
    available at the fold's `fit_origin` must still be usable by the
    adaptive model once it genuinely becomes available at some LATER
    test-year row's own cutoff -- eligibility is decided per-row by
    `safe_as_of_trailing_mean`, never by this function's own (wider)
    scope.
    """
    return frame.loc[pd.to_datetime(frame[date_col]).dt.year < test_year]


def run_point_models_for_fold(
    *, fold: Fold, cutoff_view: str, entries: list[FeatureEntry], is_provisional: bool, adaptive_candidate_pool: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Returns `(point_predictions, extras)`; `extras` carries objects
    later stages need (the fitted frozen baseline, and the point GBM's
    own selected hyperparameters) without re-fitting them."""
    cutoff_col = CUTOFF_COL_BY_VIEW[cutoff_view]
    core_cols = _core_feature_columns(entries)
    rows = []

    frozen_model = RecentHistoryBaseline(target_col=TARGET_COL, n_lookback=N_LOOKBACK).fit(fold.train)
    frozen_forecast = frozen_model.predict(fold.test)
    rows.append(
        _point_row(
            fold=fold,
            cutoff_view=cutoff_view,
            model_id="baseline_recent_history_frozen",
            forecast=frozen_forecast,
            config={"tenor_recent_means": frozen_model.tenor_recent_means_, "tenor_n_used": frozen_model.tenor_n_used_},
            is_provisional=is_provisional,
        )
    )

    adaptive_model = AdaptiveRecentHistoryBaseline(target_col=TARGET_COL, n_lookback=N_LOOKBACK).fit(
        fold.train, candidate_pool=adaptive_candidate_pool
    )
    adaptive_forecast = adaptive_model.predict(fold.test, cutoff_col=cutoff_col)
    rows.append(
        _point_row(
            fold=fold,
            cutoff_view=cutoff_view,
            model_id="baseline_recent_history_adaptive",
            forecast=adaptive_forecast,
            config={"n_lookback": N_LOOKBACK},
            is_provisional=is_provisional,
        )
    )

    gbm_result = fit_predict_gbm(fold.train, fold.test, feature_cols=core_cols, target_col=TARGET_COL)
    rows.append(
        _point_row(
            fold=fold,
            cutoff_view=cutoff_view,
            model_id="challenger_gbm_core",
            forecast=gbm_result.predictions,
            config={
                "selected_hyperparams": gbm_result.selected_hyperparams,
                "used_inner_cv": gbm_result.used_inner_cv,
                "n_feature_columns_used": len(gbm_result.kept_feature_columns),
            },
            is_provisional=is_provisional,
        )
    )

    shrinkage_model = ShrinkageTenorReopeningBaseline(target_col=TARGET_COL).fit(fold.train)
    shrinkage_forecast = shrinkage_model.predict(fold.test)
    rows.append(
        _point_row(
            fold=fold,
            cutoff_view=cutoff_view,
            model_id="challenger_shrinkage_tenor_reopening",
            forecast=shrinkage_forecast,
            config={"k_tenor": shrinkage_model.k_tenor_, "k_group": shrinkage_model.k_group_},
            is_provisional=is_provisional,
        )
    )

    predictions = pd.concat(rows, ignore_index=True)
    extras = {
        "frozen_model": frozen_model,
        "frozen_forecast": frozen_forecast,
        "gbm_selected_hyperparams": gbm_result.selected_hyperparams,
    }
    return predictions, extras


def run_probabilistic_models_for_fold(
    *, fold: Fold, cutoff_view: str, entries: list[FeatureEntry], quantile_levels: tuple[float, ...], extras: dict, is_provisional: bool
) -> pd.DataFrame:
    core_cols = _core_feature_columns(entries)
    rows = []

    calibrator = ResidualQuantileCalibrator(quantile_levels=quantile_levels, target_col=TARGET_COL).fit(
        fold.train,
        point_model_factory=lambda: RecentHistoryBaseline(target_col=TARGET_COL, n_lookback=N_LOOKBACK),
        point_predict_fn=lambda model, df: model.predict(df),
    )
    residual_quantiles = calibrator.predict_quantiles(fold.test, extras["frozen_forecast"])
    rows.append(_probabilistic_row(fold, cutoff_view, "quantile_recent_history_residual", residual_quantiles, is_provisional))

    gbm_quantiles = fit_predict_gbm_quantiles(
        fold.train,
        fold.test,
        feature_cols=core_cols,
        target_col=TARGET_COL,
        quantile_levels=quantile_levels,
        tree_hyperparams=extras["gbm_selected_hyperparams"],
    )
    rows.append(_probabilistic_row(fold, cutoff_view, "quantile_gbm_core", gbm_quantiles, is_provisional))

    return pd.concat(rows, ignore_index=True)


def _probabilistic_row(fold: Fold, cutoff_view: str, model_id: str, quantile_df: pd.DataFrame, is_provisional: bool) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            AUCTION_KEY_COL: fold.test[AUCTION_KEY_COL].to_numpy(),
            "cutoff_view": cutoff_view,
            "model_id": model_id,
            "test_year": fold.test_year,
            "is_provisional": is_provisional,
            "tenor": fold.test["tenor"].to_numpy(),
            "actual": fold.test[TARGET_COL].to_numpy(),
        }
    )
    # Same pre-declared [0, 1] clipping policy as the point forecasts
    # (configs/phase_7_protocol.yml's targets.clipping) -- a share
    # cannot be negative or exceed 1.0, and the quantile-crossing fix
    # already applied upstream means clipping cannot reintroduce
    # crossing (clip is monotone).
    for col in quantile_df.columns:
        out[col] = quantile_df[col].clip(lower=CLIP_LOWER, upper=CLIP_UPPER).to_numpy()
    return out


def run_stress_event_labels_for_fold(*, fold: Fold) -> dict:
    """Returns cross-fitted train labels/threshold, test labels/surprise,
    and the missing-training-label count for one fold. Does NOT fit the
    classifier -- that only happens after the pooled gate decision is
    known (see `phase7_cli.py`)."""
    train_label, threshold, _train_surprise = build_cross_fitted_training_labels(fold.train)
    n_missing = int(pd.to_numeric(train_label, errors="coerce").isna().sum())

    if threshold is None:
        test_label = pd.Series(pd.NA, index=fold.test.index, dtype="object")
        test_surprise = pd.Series(float("nan"), index=fold.test.index)
    else:
        test_label, test_surprise = build_test_labels(fold.train, fold.test, threshold=threshold)

    return {
        "train_label": train_label,
        "threshold": threshold,
        "test_label": test_label,
        "test_surprise": test_surprise,
        "n_missing_training_labels": n_missing,
    }


STRESS_CLASSIFIER_MODEL_ID = "stress_classifier_logistic_core"  # configs/phase_7_protocol.yml stress_event_gate.classifier.model_id


def run_stress_classifier_for_fold(
    *, fold: Fold, cutoff_view: str, entries: list[FeatureEntry], train_label: pd.Series
) -> pd.DataFrame:
    core_cols = _core_feature_columns(entries)
    train_with_label = fold.train.assign(_stress_label=train_label)

    # Acceptance-review fix: the OUTER training label's own threshold is
    # fit on cross-fitted surprises pooled across the WHOLE outer
    # training pool -- including whatever year becomes the inner-
    # validation year below. Using it directly to SCORE inner-CV
    # candidates would let that year's own surprise distribution
    # influence the threshold used to grade it. `inner_safe_label`
    # re-derives a properly nested label (see
    # `evaluation.stress_event.build_inner_cv_safe_labels`) for inner
    # hyperparameter selection ONLY; the final classifier fit and the
    # outer test-year labels below are unaffected (still `train_label`
    # / the ordinary `build_test_labels` threshold).
    inner_safe_label = build_inner_cv_safe_labels(fold.train)
    inner_label_col = None
    if inner_safe_label is not None:
        train_with_label = train_with_label.assign(_stress_label_inner_safe=inner_safe_label)
        inner_label_col = "_stress_label_inner_safe"

    result = fit_predict_logistic_classifier(
        train_with_label, fold.test, feature_cols=core_cols, label_col="_stress_label", inner_label_col=inner_label_col
    )
    return pd.DataFrame(
        {
            AUCTION_KEY_COL: fold.test[AUCTION_KEY_COL].to_numpy(),
            "cutoff_view": cutoff_view,
            "model_id": STRESS_CLASSIFIER_MODEL_ID,
            "test_year": fold.test_year,
            "predicted_proba": result.predicted_proba.to_numpy(),
            "predicted_label": result.predicted_label.to_numpy(),
            "train_prevalence": result.train_prevalence,
            "selected_c": result.selected_c,
            "used_inner_cv": result.used_inner_cv,
        }
    )


@dataclass
class Phase7Results:
    point_predictions: pd.DataFrame
    probabilistic_predictions: pd.DataFrame
    gate_decisions: dict
    classifier_predictions: pd.DataFrame
    years_excluded_from_classifier: dict


def run_full_phase7_evaluation(
    *,
    announcement_frame: pd.DataFrame,
    pre_auction_frame: pd.DataFrame,
    entries: list[FeatureEntry],
    protocol7: Protocol7,
) -> Phase7Results:
    """The single Phase 7 entry point: runs every point model, every
    probabilistic model, and the stress-event gate/classifier (only for
    a cutoff view whose gate passes), across every complete-year fold
    plus the provisional year. Mirrors Phase 6's
    `run_evaluation.run_full_evaluation` in structure and discipline.
    """
    frames = add_safe_regime_features_both_cutoffs(announcement_frame, pre_auction_frame)

    point_rows: list[pd.DataFrame] = []
    prob_rows: list[pd.DataFrame] = []
    fold_objects: dict[tuple[str, int], Fold] = {}
    stress_train_labels: dict[str, dict[int, pd.Series]] = {}
    stress_test_labels: dict[str, dict[int, pd.Series]] = {}
    stress_missing: dict[str, dict[int, int]] = {}

    for cutoff_view, frame in frames.items():
        stress_train_labels[cutoff_view] = {}
        stress_test_labels[cutoff_view] = {}
        stress_missing[cutoff_view] = {}
        for year, is_provisional in _all_fold_years(protocol7):
            fold = build_fold(frame, test_year=year)
            fold_objects[(cutoff_view, year)] = fold
            adaptive_candidate_pool = _prior_years_candidate_pool(frame, test_year=year)

            point_preds, extras = run_point_models_for_fold(
                fold=fold,
                cutoff_view=cutoff_view,
                entries=entries,
                is_provisional=is_provisional,
                adaptive_candidate_pool=adaptive_candidate_pool,
            )
            point_rows.append(point_preds)

            prob_preds = run_probabilistic_models_for_fold(
                fold=fold,
                cutoff_view=cutoff_view,
                entries=entries,
                quantile_levels=protocol7.quantile_levels,
                extras=extras,
                is_provisional=is_provisional,
            )
            prob_rows.append(prob_preds)

            stress = run_stress_event_labels_for_fold(fold=fold)
            stress_train_labels[cutoff_view][year] = stress["train_label"]
            stress_test_labels[cutoff_view][year] = stress["test_label"]
            stress_missing[cutoff_view][year] = stress["n_missing_training_labels"]

    point_predictions = pd.concat(point_rows, ignore_index=True)
    probabilistic_predictions = pd.concat(prob_rows, ignore_index=True)

    gate_decisions: dict = {}
    classifier_rows: list[pd.DataFrame] = []
    years_excluded_from_classifier: dict[str, dict[int, str]] = {}

    for cutoff_view in frames:
        complete_test_labels = {
            year: labels
            for year, labels in stress_test_labels[cutoff_view].items()
            if year in protocol7.complete_test_years
        }
        decision = evaluate_gate(
            train_labels_by_fold=stress_train_labels[cutoff_view],
            test_labels_by_fold=complete_test_labels,
            n_missing_training_labels_by_fold=stress_missing[cutoff_view],
        )
        gate_decisions[cutoff_view] = decision
        years_excluded_from_classifier[cutoff_view] = {}

        if not decision.passes:
            continue

        for year, is_provisional in _all_fold_years(protocol7):
            fold = fold_objects[(cutoff_view, year)]
            train_label = stress_train_labels[cutoff_view][year]
            try:
                clf_preds = run_stress_classifier_for_fold(
                    fold=fold, cutoff_view=cutoff_view, entries=entries, train_label=train_label
                )
            except ValueError as exc:
                # A fold whose cross-fitted training label happens to
                # have fewer than 2 distinct classes (a real data
                # characteristic of a small/early fold, not a bug)
                # cannot fit a binary classifier at all -- excluded and
                # documented explicitly, never silently skipped.
                years_excluded_from_classifier[cutoff_view][year] = str(exc)
                continue
            clf_preds["is_provisional"] = is_provisional
            clf_preds["actual_label"] = pd.Series(stress_test_labels[cutoff_view][year]).to_numpy()
            classifier_rows.append(clf_preds)

    classifier_predictions = (
        pd.concat(classifier_rows, ignore_index=True) if classifier_rows else pd.DataFrame()
    )

    return Phase7Results(
        point_predictions=point_predictions,
        probabilistic_predictions=probabilistic_predictions,
        gate_decisions=gate_decisions,
        classifier_predictions=classifier_predictions,
        years_excluded_from_classifier=years_excluded_from_classifier,
    )
