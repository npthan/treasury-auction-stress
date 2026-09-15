"""Phase 6: the evaluation driver. Ties together
`treasury_auction_stress.evaluation.{protocol,timing,data_loading}` and
`treasury_auction_stress.models.{baselines,linear_models}` into the
single out-of-sample prediction table every report/figure is generated
from.

Every model is fit exactly once per (cutoff view, fold, target) and
used to predict every auction in that fold's test year -- never
updated within the year (`configs/phase_6_evaluation.yml`'s
`frozen_annual_models`).
"""

from __future__ import annotations

import json

import pandas as pd

from treasury_auction_stress.evaluation.protocol import (
    ModelSpec,
    Protocol,
    clip_series,
    resolve_tier_feature_columns,
)
from treasury_auction_stress.evaluation.timing import (
    RESULT_SAFE_AVAILABLE_DATE_COL,
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
from treasury_auction_stress.models.baselines import (
    GlobalMeanBaseline,
    RecentHistoryBaseline,
    TenorMeanBaseline,
    TenorReopeningMeanBaseline,
)
from treasury_auction_stress.models.linear_models import fit_predict_linear

PREDICTION_COLUMNS: tuple[str, ...] = (
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
    "training_origin_date",
    "training_row_count",
    "training_max_result_safe_date",
    "test_prediction_cutoff",
    "selected_feature_tier",
    "model_config",
    "sensitivity_only",
)

CUTOFF_COL_BY_VIEW: dict[str, str] = {
    "announcement": ANNOUNCEMENT_CUTOFF_COL,
    "pre_auction": PRE_AUCTION_CUTOFF_COL,
}


def build_modeling_frame(matrix: pd.DataFrame, target_table: pd.DataFrame) -> pd.DataFrame:
    """One predictor matrix (either cutoff view) merged 1:1 with the
    target table by `auction_key`, plus `result_safe_available_date`.
    Never mutates the inputs.
    """
    merged = matrix.merge(target_table, on=AUCTION_KEY_COL, how="inner", validate="one_to_one")
    if len(merged) != len(matrix) or len(merged) != len(target_table):
        raise AssertionError("build_modeling_frame: matrix and target table key sets disagree")
    return add_result_safe_available_date(merged, date_col="auction_date")


def _fit_predict_baseline(model_spec: ModelSpec, fold: Fold, target_name: str) -> tuple[pd.Series, dict]:
    if model_spec.id == "baseline_global_mean":
        model = GlobalMeanBaseline(target_col=target_name).fit(fold.train)
        config = {"global_mean": model.global_mean_}
    elif model_spec.id == "baseline_tenor_mean":
        model = TenorMeanBaseline(target_col=target_name).fit(fold.train)
        config = {"tenor_means": model.tenor_means_, "global_mean": model.global_mean_}
    elif model_spec.id == "baseline_tenor_reopening_mean":
        model = TenorReopeningMeanBaseline(target_col=target_name).fit(fold.train)
        config = {
            "group_means": {str(k): v for k, v in model.group_means_.items()},
            "tenor_means": model.tenor_means_,
            "global_mean": model.global_mean_,
        }
    elif model_spec.id == "baseline_recent_history":
        model = RecentHistoryBaseline(
            target_col=target_name, n_lookback=model_spec.recent_history_n_lookback
        ).fit(fold.train)
        config = {"tenor_recent_means": model.tenor_recent_means_, "tenor_n_used": model.tenor_n_used_}
    else:
        raise ValueError(f"unknown baseline model id {model_spec.id!r}")
    return model.predict(fold.test), config


def _fit_predict_linear_model(
    model_spec: ModelSpec, fold: Fold, target_name: str, entries: list[FeatureEntry], protocol: Protocol
) -> tuple[pd.Series, dict, list[str]]:
    feature_cols = resolve_tier_feature_columns(protocol, model_spec.predictor_tier, entries)
    result = fit_predict_linear(
        fold.train, fold.test, feature_cols=feature_cols, target_col=target_name, estimator_type=model_spec.estimator
    )
    config = {
        "selected_hyperparams": result.selected_hyperparams,
        "used_inner_cv": result.used_inner_cv,
        "inner_cv_reason_if_not_used": result.inner_cv_reason_if_not_used,
        "dropped_all_missing_columns": result.dropped_all_missing_columns,
        "n_feature_columns_used": len(result.kept_feature_columns),
    }
    return result.predictions, config, feature_cols


def run_one_fold_model(
    *,
    model_spec: ModelSpec,
    fold: Fold,
    target_name: str,
    cutoff_view: str,
    entries: list[FeatureEntry],
    protocol: Protocol,
    is_provisional: bool,
) -> pd.DataFrame:
    """Fit `model_spec` on `fold.train` and predict `fold.test` for one
    (cutoff view, fold, target) combination. Returns one row per test
    auction, matching `PREDICTION_COLUMNS`.
    """
    if model_spec.family == "baseline":
        predictions_unclipped, config = _fit_predict_baseline(model_spec, fold, target_name)
        feature_tier = "tenor_and_reopening_only"
    elif model_spec.family == "linear":
        predictions_unclipped, config, _ = _fit_predict_linear_model(model_spec, fold, target_name, entries, protocol)
        feature_tier = model_spec.predictor_tier
    else:
        raise ValueError(f"unknown model family {model_spec.family!r}")

    predictions_unclipped = pd.Series(predictions_unclipped.to_numpy(), index=fold.test.index)
    clipped = clip_series(predictions_unclipped, target_name, protocol)
    was_clipped = (clipped != predictions_unclipped) & predictions_unclipped.notna()

    cutoff_col = CUTOFF_COL_BY_VIEW[cutoff_view]
    out = pd.DataFrame(
        {
            AUCTION_KEY_COL: fold.test[AUCTION_KEY_COL].to_numpy(),
            "cutoff_view": cutoff_view,
            "target_name": target_name,
            "model_id": model_spec.id,
            "test_year": fold.test_year,
            "fold_id": f"{fold.test_year}_provisional" if is_provisional else str(fold.test_year),
            "is_provisional": is_provisional,
            "tenor": fold.test["tenor"].to_numpy(),
            "is_reopening": fold.test["is_reopening"].to_numpy(),
            "actual": fold.test[target_name].to_numpy(),
            "forecast": clipped.to_numpy(),
            "forecast_unclipped": predictions_unclipped.to_numpy(),
            "was_clipped": was_clipped.to_numpy(),
            "training_origin_date": fold.fit_origin,
            "training_row_count": len(fold.train),
            "training_max_result_safe_date": (
                fold.train[RESULT_SAFE_AVAILABLE_DATE_COL].max() if len(fold.train) else pd.NaT
            ),
            "test_prediction_cutoff": fold.test[cutoff_col].to_numpy(),
            "selected_feature_tier": feature_tier,
            "model_config": json.dumps(config, default=str),
            "sensitivity_only": model_spec.sensitivity_only,
        }
    )
    return out[list(PREDICTION_COLUMNS)]


def run_full_evaluation(
    *,
    announcement_frame: pd.DataFrame,
    pre_auction_frame: pd.DataFrame,
    entries: list[FeatureEntry],
    protocol: Protocol,
) -> pd.DataFrame:
    """Run every pre-registered model, for every target, every cutoff
    view, every fold (complete years + the provisional year). Returns
    the full long-format out-of-sample prediction table.
    """
    frame_by_cutoff = {"announcement": announcement_frame, "pre_auction": pre_auction_frame}
    all_fold_years = [(y, False) for y in protocol.complete_test_years] + [(protocol.provisional_test_year, True)]

    rows: list[pd.DataFrame] = []
    for cutoff_view in protocol.cutoff_views:
        frame = frame_by_cutoff[cutoff_view]
        for target_name in protocol.all_targets:
            for fold_year, is_provisional in all_fold_years:
                fold = build_fold(frame, test_year=fold_year)
                for model_spec in protocol.models:
                    result = run_one_fold_model(
                        model_spec=model_spec,
                        fold=fold,
                        target_name=target_name,
                        cutoff_view=cutoff_view,
                        entries=entries,
                        protocol=protocol,
                        is_provisional=is_provisional,
                    )
                    rows.append(result)

    predictions = pd.concat(rows, ignore_index=True)

    dedupe_key = [AUCTION_KEY_COL, "model_id", "cutoff_view", "target_name"]
    if predictions.duplicated(subset=dedupe_key).any():
        dupes = predictions.loc[predictions.duplicated(subset=dedupe_key, keep=False), dedupe_key]
        raise AssertionError(f"run_full_evaluation: duplicate predictions found:\n{dupes.drop_duplicates().head(10)}")

    return predictions
