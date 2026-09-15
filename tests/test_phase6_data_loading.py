from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from treasury_auction_stress.evaluation.data_loading import (
    DEFAULT_PROCESSED_DIR,
    REQUIRED_FILES,
    load_phase6_inputs,
    validate_phase6_inputs,
)


def _skip_if_no_processed_data():
    if not (DEFAULT_PROCESSED_DIR / "feature_matrix_announcement.parquet").exists():
        pytest.skip("processed Phase 5 tables not present in this environment")


def test_load_phase6_inputs_missing_file_raises_with_regeneration_command(tmp_path: Path):
    empty_dir = tmp_path / "processed"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        load_phase6_inputs(empty_dir)
    assert "feature_matrix_cli" in str(exc_info.value)


def test_required_files_list_matches_six_phase5_artifacts():
    assert len(REQUIRED_FILES) == 6


def test_load_and_validate_real_phase6_inputs():
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    report = validate_phase6_inputs(inputs)
    assert report["n_rows"] == len(inputs.announcement_matrix)
    assert len(inputs.fingerprints) == 6
    for family, results in report["availability_results"]["announcement"].items():
        assert results["n_violations"] == 0, f"{family} has a timing violation"
    for family, results in report["availability_results"]["pre_auction"].items():
        assert results["n_violations"] == 0, f"{family} has a timing violation"


def test_validate_phase6_inputs_rejects_a_forbidden_column_injected_into_the_matrix():
    """The required 'intentionally leaky implementation ... that the
    validator rejects' scenario: inject a real result-side column into
    a copy of the real announcement matrix and prove validation now
    raises where it previously passed."""
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    poisoned = inputs.announcement_matrix.copy()
    poisoned["primary_dealer_accepted"] = 999_999.0  # a forbidden result column

    from treasury_auction_stress.evaluation.data_loading import Phase6Inputs

    poisoned_inputs = Phase6Inputs(
        announcement_matrix=poisoned,
        pre_auction_matrix=inputs.pre_auction_matrix,
        target_table=inputs.target_table,
        announcement_audit=inputs.announcement_audit,
        pre_auction_audit=inputs.pre_auction_audit,
        pre_auction_update_table=inputs.pre_auction_update_table,
        fingerprints=inputs.fingerprints,
    )
    with pytest.raises(AssertionError):
        validate_phase6_inputs(poisoned_inputs)


def test_validate_phase6_inputs_rejects_missing_primary_target():
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    poisoned_targets = inputs.target_table.copy()
    poisoned_targets.loc[poisoned_targets.index[0], "primary_dealer_share"] = pd.NA

    from treasury_auction_stress.evaluation.data_loading import Phase6Inputs

    poisoned_inputs = Phase6Inputs(
        announcement_matrix=inputs.announcement_matrix,
        pre_auction_matrix=inputs.pre_auction_matrix,
        target_table=poisoned_targets,
        announcement_audit=inputs.announcement_audit,
        pre_auction_audit=inputs.pre_auction_audit,
        pre_auction_update_table=inputs.pre_auction_update_table,
        fingerprints=inputs.fingerprints,
    )
    with pytest.raises(AssertionError):
        validate_phase6_inputs(poisoned_inputs)


def test_validate_phase6_inputs_rejects_key_set_disagreement():
    _skip_if_no_processed_data()
    inputs = load_phase6_inputs()
    truncated_targets = inputs.target_table.iloc[1:].copy()

    from treasury_auction_stress.evaluation.data_loading import Phase6Inputs

    poisoned_inputs = Phase6Inputs(
        announcement_matrix=inputs.announcement_matrix,
        pre_auction_matrix=inputs.pre_auction_matrix,
        target_table=truncated_targets,
        announcement_audit=inputs.announcement_audit,
        pre_auction_audit=inputs.pre_auction_audit,
        pre_auction_update_table=inputs.pre_auction_update_table,
        fingerprints=inputs.fingerprints,
    )
    with pytest.raises(AssertionError):
        validate_phase6_inputs(poisoned_inputs)
