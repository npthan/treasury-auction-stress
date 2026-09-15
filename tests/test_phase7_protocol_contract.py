"""Phase 7 acceptance review, Gate 2: exhaustive contract checks between
`configs/phase_7_protocol.yml` (the frozen decision record) and the
actual hardcoded constants in every Phase 7 implementation module.

`treasury_auction_stress.evaluation.protocol7.load_protocol7` only
reads a SMALL subset of the YAML at runtime (fold schedule, primary
target, quantile levels) -- exactly like Phase 6's own
`evaluation.protocol.load_protocol` only reads its own subset (model
list, targets, clipping, predictor tiers) and leaves grids like
`RIDGE_ALPHA_GRID` as separate, hardcoded module constants. Every OTHER
Phase 7 decision (the GBM grid, the stress percentile, the classifier's
C grid, the two data-sufficiency count thresholds, the quantile
residual pooling-fallback threshold, and every declared model_id) is a
hardcoded Python constant, matching Phase 6's own precedent
(`tests/test_phase6_protocol.py` is the established pattern this file
mirrors: it does not make the code re-read the YAML for everything --
it makes MISMATCHES between the YAML and the code fail loudly).

Every test below fails LOUDLY if a code constant and the YAML's own
declared value ever disagree -- this is what "genuinely drive those
decisions, or add exhaustive, failing-on-mismatch contract checks"
means in practice for values that are legitimately implemented as
plain Python constants (for performance, simplicity, or because
`configs/phase_6_evaluation.yml` already established this pattern).
"""

from __future__ import annotations

from itertools import product

import pytest
import yaml

from treasury_auction_stress.evaluation import (
    reporting7,
    run_evaluation_phase7,
    stress_event,
)
from treasury_auction_stress.evaluation.protocol7 import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol7,
)
from treasury_auction_stress.models import (
    adaptive_baselines,
    gbm,
    logistic_classifier,
    quantile_residual,
)


@pytest.fixture(scope="module")
def raw_yaml() -> dict:
    return yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_primary_target_matches_yaml(raw_yaml):
    assert run_evaluation_phase7.TARGET_COL == raw_yaml["targets"]["primary"]
    assert stress_event.TARGET_COL == raw_yaml["targets"]["primary"]


def test_clipping_bounds_match_yaml(raw_yaml):
    clipping = raw_yaml["targets"]["clipping"]
    assert run_evaluation_phase7.CLIP_LOWER == clipping["lower"]
    assert run_evaluation_phase7.CLIP_UPPER == clipping["upper"]


def test_cutoff_views_match_yaml(raw_yaml):
    assert set(run_evaluation_phase7.CUTOFF_COL_BY_VIEW.keys()) == set(raw_yaml["cutoff_views"])


def test_predictor_tier_matches_yaml(raw_yaml):
    assert run_evaluation_phase7.GBM_FEATURE_TIER == raw_yaml["predictor_tier"]["name"]


def test_adaptive_n_lookback_matches_yaml(raw_yaml):
    declared = raw_yaml["adaptive_recent_history"]["n_lookback"]
    assert run_evaluation_phase7.N_LOOKBACK == declared
    assert adaptive_baselines.DEFAULT_N_LOOKBACK == declared


def test_model_ids_match_yaml_declared_ids(raw_yaml):
    frozen, adaptive, gbm_id, shrinkage_id = run_evaluation_phase7.POINT_MODEL_IDS
    assert adaptive == raw_yaml["adaptive_recent_history"]["model_id"]
    assert frozen == raw_yaml["adaptive_recent_history"]["frozen_counterpart_model_id"]
    assert gbm_id == raw_yaml["gbm_challenger"]["model_id"]
    assert shrinkage_id == raw_yaml["shrinkage_challenger"]["model_id"]


def test_probabilistic_model_ids_match_yaml(raw_yaml):
    residual_id, gbm_quantile_id = run_evaluation_phase7.PROBABILISTIC_MODEL_IDS
    assert residual_id == raw_yaml["probabilistic"]["conservative_baseline_method"]["model_id"]
    assert gbm_quantile_id == raw_yaml["probabilistic"]["prespecified_challenger"]["model_id"]


def test_stress_classifier_model_id_matches_yaml(raw_yaml):
    assert run_evaluation_phase7.STRESS_CLASSIFIER_MODEL_ID == raw_yaml["stress_event_gate"]["classifier"]["model_id"]


def test_reporting7_baseline_and_challenger_subsets_cover_exactly_point_model_ids():
    """`reporting7.FROZEN_BASELINE`/`ADAPTIVE_BASELINE`/`CHALLENGERS` split
    `POINT_MODEL_IDS` into named subsets for report rendering -- this
    proves they are still an exact, non-overlapping partition of it,
    never silently missing or duplicating a model as either list
    changes over time.
    """
    reconstructed = {reporting7.FROZEN_BASELINE, reporting7.ADAPTIVE_BASELINE, *reporting7.CHALLENGERS}
    assert reconstructed == set(run_evaluation_phase7.POINT_MODEL_IDS)
    assert len(reconstructed) == len(run_evaluation_phase7.POINT_MODEL_IDS)


def test_gbm_grid_matches_yaml_regularization_constraints(raw_yaml):
    constraints = raw_yaml["gbm_challenger"]["regularization_constraints"]
    expected_grid = {
        frozenset(
            {
                "max_iter": max_iter,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "max_leaf_nodes": constraints["max_leaf_nodes"],
                "min_samples_leaf": constraints["min_samples_leaf"],
                "l2_regularization": constraints["l2_regularization"],
                "random_state": constraints["random_state"],
            }.items()
        )
        for max_iter, max_depth, learning_rate in product(
            constraints["max_iter"], constraints["max_depth"], constraints["learning_rate"]
        )
    }
    actual_grid = {frozenset(candidate.items()) for candidate in gbm.GBM_HYPERPARAM_GRID}
    assert actual_grid == expected_grid
    assert len(gbm.GBM_HYPERPARAM_GRID) == raw_yaml["gbm_challenger"]["grid_size"]


def test_gbm_fallback_matches_yaml(raw_yaml):
    assert gbm.FALLBACK_GBM_HYPERPARAMS == raw_yaml["gbm_challenger"]["fallback_if_insufficient_history"]


def test_quantile_levels_match_yaml(raw_yaml):
    protocol7 = load_protocol7()
    assert list(protocol7.quantile_levels) == raw_yaml["probabilistic"]["quantile_levels"]


def test_nominal_intervals_match_yaml(raw_yaml):
    from treasury_auction_stress.evaluation.probabilistic_metrics import (
        NOMINAL_INTERVALS,
    )

    declared = raw_yaml["probabilistic"]["nominal_intervals_reported"]
    # Each declared string is e.g. "80% (q10-q90)" -- parse the label and
    # the two quantile levels out of it and compare against the code's
    # own (label, low_q, high_q) tuples, rather than string-matching the
    # whole sentence (which would be brittle to punctuation changes).
    parsed = []
    for entry in declared:
        label, rest = entry.split(" (", 1)
        q_part = rest.rstrip(")")
        low_str, high_str = q_part[1:].split("-q")
        parsed.append((label, float(f"0.{low_str}"), float(f"0.{high_str}")))
    assert parsed == list(NOMINAL_INTERVALS)


def test_quantile_residual_pooling_fallback_matches_yaml(raw_yaml):
    declared = raw_yaml["probabilistic"]["conservative_baseline_method"]["min_residuals_per_tenor_fallback_threshold"]
    assert quantile_residual.MIN_RESIDUALS_PER_TENOR == declared


def test_stress_percentile_matches_yaml(raw_yaml):
    assert stress_event.STRESS_PERCENTILE == raw_yaml["stress_event_gate"]["threshold_percentile"]


def test_stress_event_count_thresholds_match_yaml(raw_yaml):
    assert stress_event.MIN_POOLED_POSITIVE_COUNT == raw_yaml["stress_event_gate"]["min_pooled_positive_count"]
    assert stress_event.MIN_YEAR_POSITIVE_COUNT == raw_yaml["stress_event_gate"]["min_year_positive_count"]


def test_classifier_c_grid_matches_yaml(raw_yaml):
    declared = raw_yaml["stress_event_gate"]["classifier"]["hyperparameter_grid"]["C"]
    assert list(logistic_classifier.C_GRID) == declared


def test_safe_regime_window_matches_adaptive_lookback_and_yaml(raw_yaml):
    """The stress-event gate's safe regime feature and the adaptive
    recent-history benchmark both use an 8-auction window -- verified
    to be the SAME declared value, not two coincidentally-equal
    literals.
    """
    assert stress_event.REGIME_WINDOW == raw_yaml["adaptive_recent_history"]["n_lookback"]


def test_every_gbm_config_and_fallback_uses_the_declared_random_state(raw_yaml):
    declared_seed = raw_yaml["gbm_challenger"]["regularization_constraints"]["random_state"]
    for candidate in gbm.GBM_HYPERPARAM_GRID:
        assert candidate["random_state"] == declared_seed
    assert gbm.FALLBACK_GBM_HYPERPARAMS["random_state"] == declared_seed
