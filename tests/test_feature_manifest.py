from __future__ import annotations

import pytest

from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    assert_predictor_columns_match_contract,
    literal_predictor_names,
    literal_target_names,
    load_contract,
    parse_entries,
    render_feature_dictionary_markdown,
)


@pytest.fixture(scope="module")
def contract():
    return load_contract(DEFAULT_CONTRACT_PATH)


@pytest.fixture(scope="module")
def entries(contract):
    return parse_entries(contract)


def test_contract_file_exists():
    assert DEFAULT_CONTRACT_PATH.exists()


def test_every_entry_is_valid(entries):
    assert len(entries) > 0
    for e in entries:
        assert e.feature_role in {
            "identifier", "cutoff_metadata", "predictor", "audit_provenance",
            "target", "target_diagnostic", "target_construction_only", "excluded",
        }
        assert e.default_tier in {"core", "extended", "audit_only", "excluded"}


def test_every_main_matrix_predictor_entry_is_literal_not_pattern(entries):
    """Every predictor that belongs to one of the two Phase 5 predictor
    matrices must be named literally, so the contract and the actual
    matrix columns can be cross-checked exactly. The small
    pre_auction_incremental family is exempt -- it lives in a separate,
    optional update table (data/processed/pre_auction_information_
    updates.parquet), not either predictor matrix, and its columns are
    checked by name directly in tests/test_feature_matrix.py instead.
    """
    for e in entries:
        if e.feature_role == "predictor":
            assert e.default_tier in ("core", "extended")
            if e.fields.get("source_family") != "pre_auction_incremental":
                assert not e.is_pattern, f"{e.name} is a predictor but only given as a name_pattern"


def test_literal_predictor_names_nonempty(entries):
    predictors = literal_predictor_names(entries)
    assert len(predictors) > 40  # comprehensive across all 5 source families
    assert "tenor" in predictors
    assert predictors["tenor"] == "core"
    assert "dealer_net_position_total_ex_tips_harmonized" in predictors
    assert "matched_tenor_par_yield_percent" in predictors


def test_literal_target_names(entries):
    targets = literal_target_names(entries)
    assert targets["primary_dealer_share"] == "target"
    assert targets["direct_bidder_share"] == "target"
    assert targets["indirect_bidder_share"] == "target"
    assert targets["bid_to_cover_ratio"] == "target"
    assert targets["noncompetitive_share"] == "target_diagnostic"


def test_no_target_or_diagnostic_field_is_ever_tiered_as_a_predictor(entries):
    for e in entries:
        if e.feature_role in ("target", "target_diagnostic", "target_construction_only", "excluded"):
            assert e.default_tier != "core" and e.default_tier != "extended"


def test_assert_predictor_columns_match_contract_passes_on_matching_set(entries):
    predictors = literal_predictor_names(entries)
    assert_predictor_columns_match_contract(list(predictors), entries)


def test_assert_predictor_columns_match_contract_fails_on_extra_column(entries):
    predictors = literal_predictor_names(entries)
    columns = [*predictors, "some_leaked_result_field"]
    with pytest.raises(AssertionError):
        assert_predictor_columns_match_contract(columns, entries)


def test_assert_predictor_columns_match_contract_fails_on_missing_column(entries):
    predictors = literal_predictor_names(entries)
    columns = list(predictors)[:-1]
    with pytest.raises(AssertionError):
        assert_predictor_columns_match_contract(columns, entries)


def test_render_feature_dictionary_markdown_contains_every_predictor(contract, entries):
    md = render_feature_dictionary_markdown(contract, entries)
    assert "# Phase 5 Feature Dictionary" in md
    for name in literal_predictor_names(entries):
        assert f"`{name}`" in md
