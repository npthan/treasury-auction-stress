"""Integration tests for the Phase 6 evaluation driver, against real
Phase 5 data when present (skipped otherwise, same convention as
`tests/test_cftc_profile_cli.py`). The full evaluation is run ONCE per
test session (module-scoped fixture) and reused across assertions --
these are read-only checks on its output, never a second full run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.data_loading import (
    DEFAULT_PROCESSED_DIR,
    load_phase6_inputs,
    validate_phase6_inputs,
)
from treasury_auction_stress.evaluation.metrics import r2_vs_strong_baseline
from treasury_auction_stress.evaluation.protocol import load_protocol
from treasury_auction_stress.evaluation.run_evaluation import (
    build_modeling_frame,
    run_full_evaluation,
)
from treasury_auction_stress.evaluation.timing import build_fold
from treasury_auction_stress.features.feature_manifest import (
    load_contract,
    parse_entries,
)
from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL


def _skip_if_no_processed_data():
    if not (DEFAULT_PROCESSED_DIR / "feature_matrix_announcement.parquet").exists():
        pytest.skip("processed Phase 5 tables not present in this environment")


@pytest.fixture(scope="module")
def real_predictions() -> pd.DataFrame:
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    validate_phase6_inputs(inputs)
    protocol = load_protocol()
    entries = parse_entries(load_contract())
    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)
    return run_full_evaluation(
        announcement_frame=ann_frame, pre_auction_frame=pre_frame, entries=entries, protocol=protocol
    )


@pytest.fixture(scope="module")
def real_frames():
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)
    return ann_frame, pre_frame


def test_exactly_one_prediction_per_auction_model_cutoff_target(real_predictions):
    dedupe_key = [AUCTION_KEY_COL, "model_id", "cutoff_view", "target_name"]
    assert not real_predictions.duplicated(subset=dedupe_key).any()


def test_no_missing_forecasts(real_predictions):
    assert real_predictions["forecast"].notna().all()
    assert real_predictions["actual"].notna().all()


def test_identical_test_key_sets_across_cutoff_views(real_predictions):
    for (model_id, target_name, fold_id), group in real_predictions.groupby(["model_id", "target_name", "fold_id"]):
        ann_keys = set(group.loc[group["cutoff_view"] == "announcement", AUCTION_KEY_COL])
        pre_keys = set(group.loc[group["cutoff_view"] == "pre_auction", AUCTION_KEY_COL])
        assert ann_keys == pre_keys, f"{model_id}/{target_name}/{fold_id}: cutoff views test different auctions"


def test_no_train_test_overlap_for_every_real_fold(real_frames):
    ann_frame, pre_frame = real_frames
    protocol = load_protocol()
    all_fold_years = list(protocol.complete_test_years) + [protocol.provisional_test_year]
    for frame in (ann_frame, pre_frame):
        for year in all_fold_years:
            fold = build_fold(frame, test_year=year)
            overlap = set(fold.train[AUCTION_KEY_COL]) & set(fold.test[AUCTION_KEY_COL])
            assert not overlap, f"year {year}: train/test overlap {overlap}"


def test_frozen_annual_model_unaffected_by_test_year_outcomes(real_frames):
    """Poison every target value in a fold's TEST set (never touching
    training data) and prove the fitted model's predictions for that
    fold are completely unchanged -- the defining property of a frozen
    annual model."""
    ann_frame, _ = real_frames
    from treasury_auction_stress.models.baselines import TenorReopeningMeanBaseline

    fold = build_fold(ann_frame, test_year=2020)
    baseline = TenorReopeningMeanBaseline(target_col="primary_dealer_share").fit(fold.train)
    before = baseline.predict(fold.test)

    poisoned_test = fold.test.copy()
    poisoned_test["primary_dealer_share"] = 99.0  # test-year outcomes must never matter
    after = baseline.predict(poisoned_test)
    pd.testing.assert_series_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_no_training_row_has_a_test_year_outcome_baked_in(real_frames):
    """A stronger poisoning test at the fold-construction level: setting
    every TEST row's target to an extreme value must never change which
    rows land in TRAIN (fold construction depends only on dates/
    availability, never on target values) nor the model fit on those
    unaffected training rows."""
    ann_frame, _ = real_frames
    from treasury_auction_stress.models.baselines import GlobalMeanBaseline

    fold = build_fold(ann_frame, test_year=2019)
    baseline = GlobalMeanBaseline(target_col="primary_dealer_share").fit(fold.train)

    poisoned_frame = ann_frame.copy()
    test_mask = poisoned_frame[AUCTION_KEY_COL].isin(fold.test[AUCTION_KEY_COL])
    poisoned_frame.loc[test_mask, "primary_dealer_share"] = -999.0
    poisoned_fold = build_fold(poisoned_frame, test_year=2019)
    poisoned_baseline = GlobalMeanBaseline(target_col="primary_dealer_share").fit(poisoned_fold.train)

    assert baseline.global_mean_ == pytest.approx(poisoned_baseline.global_mean_)
    assert sorted(fold.train[AUCTION_KEY_COL]) == sorted(poisoned_fold.train[AUCTION_KEY_COL])


def test_clipping_policy_applied_and_recorded(real_predictions):
    share_targets = {"primary_dealer_share", "direct_bidder_share", "indirect_bidder_share"}
    shares = real_predictions.loc[real_predictions["target_name"].isin(share_targets)]
    assert (shares["forecast"] >= -1e-9).all()
    assert (shares["forecast"] <= 1.0 + 1e-9).all()
    clipped_rows = shares.loc[shares["was_clipped"]]
    if not clipped_rows.empty:
        assert (
            (clipped_rows["forecast_unclipped"] < 0) | (clipped_rows["forecast_unclipped"] > 1)
        ).all()

    ratio = real_predictions.loc[real_predictions["target_name"] == "bid_to_cover_ratio"]
    assert not ratio["was_clipped"].any()  # bid_to_cover_ratio has no declared clip bound


def test_benchmark_relative_r2_uses_identical_rows_for_numerator_and_denominator(real_predictions):
    subset = real_predictions.loc[
        (real_predictions["target_name"] == "primary_dealer_share")
        & (real_predictions["cutoff_view"] == "announcement")
        & (~real_predictions["is_provisional"])
    ]
    model_df = subset.loc[subset["model_id"] == "core_ridge", ["auction_key", "actual", "forecast"]]
    baseline_df = subset.loc[
        subset["model_id"] == "baseline_tenor_reopening_mean", ["auction_key", "actual", "forecast"]
    ]
    result = r2_vs_strong_baseline(model_df, baseline_df)
    assert result is not None
    # A mismatched subset (dropping one row) must raise, not silently compute on unmatched rows.
    with pytest.raises(AssertionError):
        r2_vs_strong_baseline(model_df.iloc[1:], baseline_df)


def test_sensitivity_model_never_uses_pre_auction_incremental_columns(real_predictions):
    sensitivity_rows = real_predictions.loc[real_predictions["model_id"] == "extended_ridge_sensitivity"]
    assert not sensitivity_rows.empty
    assert sensitivity_rows["sensitivity_only"].all()


def test_training_row_count_matches_independently_built_fold(real_frames):
    ann_frame, _ = real_frames
    fold = build_fold(ann_frame, test_year=2018)
    from treasury_auction_stress.evaluation.protocol import load_protocol
    from treasury_auction_stress.evaluation.run_evaluation import run_one_fold_model
    from treasury_auction_stress.features.feature_manifest import (
        load_contract,
        parse_entries,
    )

    protocol = load_protocol()
    entries = parse_entries(load_contract())
    model_spec = next(m for m in protocol.models if m.id == "baseline_global_mean")
    result = run_one_fold_model(
        model_spec=model_spec,
        fold=fold,
        target_name="primary_dealer_share",
        cutoff_view="announcement",
        entries=entries,
        protocol=protocol,
        is_provisional=False,
    )
    assert (result["training_row_count"] == len(fold.train)).all()
    assert len(fold.train) < len(ann_frame)  # never the full dataset


def test_dropped_all_missing_column_recorded_for_early_folds(real_predictions):
    import json

    early = real_predictions.loc[
        (real_predictions["model_id"] == "core_ridge")
        & (real_predictions["cutoff_view"] == "announcement")
        & (real_predictions["target_name"] == "primary_dealer_share")
        & (real_predictions["test_year"] == 2015)
    ]
    config = json.loads(early["model_config"].iloc[0])
    assert "dealer_long_short_imbalance" in config["dropped_all_missing_columns"]
