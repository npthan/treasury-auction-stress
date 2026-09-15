from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.protocol import (
    clip_series,
    load_protocol,
    resolve_tier_feature_columns,
)
from treasury_auction_stress.features.feature_manifest import (
    load_contract,
    parse_entries,
)


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


@pytest.fixture(scope="module")
def entries():
    return parse_entries(load_contract())


def test_protocol_targets(protocol):
    assert protocol.primary_target == "primary_dealer_share"
    assert protocol.secondary_targets == ("direct_bidder_share", "indirect_bidder_share", "bid_to_cover_ratio")
    assert protocol.all_targets == (
        "primary_dealer_share",
        "direct_bidder_share",
        "indirect_bidder_share",
        "bid_to_cover_ratio",
    )


def test_protocol_strong_baseline(protocol):
    assert protocol.strong_baseline_model_id == "baseline_tenor_reopening_mean"


def test_protocol_fold_schedule(protocol):
    assert protocol.complete_test_years == tuple(range(2015, 2026))
    assert protocol.provisional_test_year == 2026


def test_structural_only_tier_has_exactly_nine_columns(protocol, entries):
    cols = resolve_tier_feature_columns(protocol, "structural_only", entries)
    assert len(cols) == 9
    assert set(cols) == {
        "tenor",
        "is_reopening",
        "offering_amt",
        "log_offering_amt",
        "days_announcement_to_auction",
        "auction_month",
        "auction_quarter",
        "auction_day_of_week",
        "announcement_day_of_week",
    }


def test_core_tier_matches_phase5_core_count(protocol, entries):
    cols = resolve_tier_feature_columns(protocol, "core", entries)
    assert len(cols) == 36


def test_extended_sensitivity_tier_is_core_plus_extended(protocol, entries):
    core = set(resolve_tier_feature_columns(protocol, "core", entries))
    extended_sensitivity = set(resolve_tier_feature_columns(protocol, "extended_sensitivity", entries))
    assert core.issubset(extended_sensitivity)
    assert len(extended_sensitivity) == 71
    assert "cutoff_gap_calendar_days" not in extended_sensitivity  # pre_auction_incremental, never a main predictor


def test_resolve_tier_feature_columns_raises_on_unknown_tier(protocol, entries):
    with pytest.raises(ValueError):
        resolve_tier_feature_columns(protocol, "not_a_tier", entries)


def test_clip_series_share_target(protocol):
    values = pd.Series([-0.1, 0.5, 1.2])
    clipped = clip_series(values, "primary_dealer_share", protocol)
    assert clipped.tolist() == [0.0, 0.5, 1.0]


def test_clip_series_bid_to_cover_has_no_upper_bound(protocol):
    values = pd.Series([-0.5, 2.0, 500.0])
    clipped = clip_series(values, "bid_to_cover_ratio", protocol)
    assert clipped.tolist() == [0.0, 2.0, 500.0]


def test_models_include_all_seven_pre_registered_plus_sensitivity(protocol):
    ids = {m.id for m in protocol.models}
    required = {
        "baseline_global_mean",
        "baseline_tenor_mean",
        "baseline_tenor_reopening_mean",
        "baseline_recent_history",
        "structural_ridge",
        "core_ridge",
        "core_elastic_net",
        "extended_ridge_sensitivity",
    }
    assert required <= ids
    sensitivity_ids = {m.id for m in protocol.models if m.sensitivity_only}
    assert sensitivity_ids == {"extended_ridge_sensitivity"}
