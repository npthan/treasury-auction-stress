"""Phase 7: real-data integration tests for
`treasury_auction_stress.evaluation.run_evaluation_phase7`.

Restricted to a 2-year test-year subset (via a locally-constructed
`Protocol7`, never a monkeypatch of global state) purely to keep this
test's wall-clock time reasonable -- the fold-construction and
leakage-relevant machinery underneath is identical regardless of how
many test years are requested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from treasury_auction_stress.evaluation.data_loading import (
    load_phase6_inputs,
    validate_phase6_inputs,
)
from treasury_auction_stress.evaluation.protocol7 import Protocol7, load_protocol7
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    POINT_MODEL_IDS,
    build_modeling_frame,
    run_full_phase7_evaluation,
)
from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
    parse_entries,
)

REQUIRED_FILES = (
    "data/processed/feature_matrix_announcement.parquet",
    "data/processed/feature_matrix_pre_auction.parquet",
    "data/processed/auction_targets.parquet",
    "data/processed/feature_audit_announcement.parquet",
    "data/processed/feature_audit_pre_auction.parquet",
    "data/processed/pre_auction_information_updates.parquet",
)


def _skip_if_processed_data_missing():
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        pytest.skip(f"Phase 5 processed artifacts not present in this environment: {missing}")


@pytest.fixture(scope="module")
def small_phase7_results():
    _skip_if_processed_data_missing()
    inputs = load_phase6_inputs()
    validate_phase6_inputs(inputs)
    entries = parse_entries(load_contract(DEFAULT_CONTRACT_PATH))
    full_protocol7 = load_protocol7()

    small_protocol7 = Protocol7(
        raw=full_protocol7.raw,
        primary_target=full_protocol7.primary_target,
        complete_test_years=(2015, 2016),
        provisional_test_year=full_protocol7.provisional_test_year,
        initial_training_start=full_protocol7.initial_training_start,
        initial_training_end=full_protocol7.initial_training_end,
        quantile_levels=full_protocol7.quantile_levels,
    )

    ann_frame = build_modeling_frame(inputs.announcement_matrix, inputs.target_table)
    pre_frame = build_modeling_frame(inputs.pre_auction_matrix, inputs.target_table)
    return run_full_phase7_evaluation(
        announcement_frame=ann_frame, pre_auction_frame=pre_frame, entries=entries, protocol7=small_protocol7
    )


def test_point_predictions_have_unique_keys_per_model_and_cutoff(small_phase7_results):
    preds = small_phase7_results.point_predictions
    dupe_key = ["auction_key", "model_id", "cutoff_view"]
    assert not preds.duplicated(subset=dupe_key).any()


def test_point_predictions_are_finite_and_within_clip_bounds(small_phase7_results):
    preds = small_phase7_results.point_predictions
    assert np.isfinite(preds["forecast"].to_numpy()).all()
    assert (preds["forecast"] >= 0.0).all()
    assert (preds["forecast"] <= 1.0).all()
    assert preds["actual"].notna().all()


def test_every_model_covers_the_same_auction_keys(small_phase7_results):
    preds = small_phase7_results.point_predictions
    key_sets = {model_id: set(preds.loc[preds["model_id"] == model_id, "auction_key"]) for model_id in POINT_MODEL_IDS}
    first = next(iter(key_sets.values()))
    for model_id, keys in key_sets.items():
        assert keys == first, f"{model_id} covers a different auction-key set than other Phase 7 models"


def test_announcement_and_pre_auction_cutoffs_cover_identical_auctions(small_phase7_results):
    preds = small_phase7_results.point_predictions
    ann_keys = set(preds.loc[preds["cutoff_view"] == "announcement", "auction_key"])
    pre_keys = set(preds.loc[preds["cutoff_view"] == "pre_auction", "auction_key"])
    assert ann_keys == pre_keys


def test_probabilistic_quantiles_are_non_decreasing_row_wise(small_phase7_results):
    prob = small_phase7_results.probabilistic_predictions
    quantile_cols = [c for c in prob.columns if c.startswith("q0.")]
    values = prob[quantile_cols].to_numpy()
    assert (np.diff(values, axis=1) >= -1e-9).all()
    assert (values >= 0.0).all() and (values <= 1.0).all()


def test_gate_decision_counts_are_internally_consistent(small_phase7_results):
    for decision in small_phase7_results.gate_decisions.values():
        assert decision.pooled_positive_count <= decision.pooled_labeled_count
        assert sum(decision.per_year_positive_counts.values()) == decision.pooled_positive_count


def test_classifier_predictions_only_exist_for_cutoffs_whose_gate_passed(small_phase7_results):
    clf = small_phase7_results.classifier_predictions
    for cutoff_view, decision in small_phase7_results.gate_decisions.items():
        clf_for_view = clf.loc[clf["cutoff_view"] == cutoff_view] if not clf.empty else clf
        if not decision.passes:
            assert clf_for_view.empty


def test_no_forbidden_result_column_leaks_into_the_core_feature_tier():
    """A cross-cutting Step-4 check: the core predictor tier used by
    every Phase 7 point/probabilistic/classifier model must never
    contain a post-auction result field."""
    from treasury_auction_stress.evaluation.protocol import (
        load_protocol,
        resolve_tier_feature_columns,
    )
    from treasury_auction_stress.features.leakage_audit import forbidden_field_scan

    entries = parse_entries(load_contract(DEFAULT_CONTRACT_PATH))
    protocol6 = load_protocol()
    core_cols = resolve_tier_feature_columns(protocol6, "core", entries)
    scan = forbidden_field_scan(core_cols)
    assert not scan["exact_matches"]
    assert not scan["pattern_matches"]
