"""Phase 6: load and validate the Phase 5 artifacts this project's
evaluation is built on top of. Nothing here recomputes a Phase 5 join
or rebuilds a matrix differently -- if a prerequisite file is missing,
`load_phase6_inputs` raises with the exact Phase 5 regeneration
command, rather than silently downloading or reconstructing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
    parse_entries,
)
from treasury_auction_stress.features.feature_matrix import (
    AUCTION_KEY_COL,
    CUTOFF_COL_BY_NAME,
    assert_audit_table_availability_safe,
    compute_content_digest,
    validate_audit_table,
    validate_pre_auction_update_table,
    validate_predictor_matrix,
    validate_target_table,
)
from treasury_auction_stress.features.leakage_audit import forbidden_field_scan

DEFAULT_PROCESSED_DIR = Path("data/processed")

REQUIRED_FILES: tuple[str, ...] = (
    "feature_matrix_announcement.parquet",
    "feature_matrix_pre_auction.parquet",
    "auction_targets.parquet",
    "feature_audit_announcement.parquet",
    "feature_audit_pre_auction.parquet",
    "pre_auction_information_updates.parquet",
)

REGENERATION_COMMAND = "uv run python -m treasury_auction_stress.features.feature_matrix_cli"


@dataclass(frozen=True)
class Phase6Inputs:
    announcement_matrix: pd.DataFrame
    pre_auction_matrix: pd.DataFrame
    target_table: pd.DataFrame
    announcement_audit: pd.DataFrame
    pre_auction_audit: pd.DataFrame
    pre_auction_update_table: pd.DataFrame
    fingerprints: dict[str, str]


def _require_files(processed_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 6: missing required Phase 5 artifact(s) in "
            f"{processed_dir}: {missing}. Regenerate Phase 5 first with:\n"
            f"    {REGENERATION_COMMAND}\n"
            "Phase 6 never rebuilds these differently or downloads new data itself."
        )


def load_phase6_inputs(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Phase6Inputs:
    """Load all six required Phase 5 artifacts and compute a content
    fingerprint for each -- never re-derive them differently. Call
    `validate_phase6_inputs` on the result before using it for
    modeling.
    """
    _require_files(processed_dir)

    announcement_matrix = pd.read_parquet(processed_dir / "feature_matrix_announcement.parquet")
    pre_auction_matrix = pd.read_parquet(processed_dir / "feature_matrix_pre_auction.parquet")
    target_table = pd.read_parquet(processed_dir / "auction_targets.parquet")
    announcement_audit = pd.read_parquet(processed_dir / "feature_audit_announcement.parquet")
    pre_auction_audit = pd.read_parquet(processed_dir / "feature_audit_pre_auction.parquet")
    pre_auction_update_table = pd.read_parquet(processed_dir / "pre_auction_information_updates.parquet")

    fingerprints = {
        name: compute_content_digest(pd.read_parquet(processed_dir / name)) for name in REQUIRED_FILES
    }

    return Phase6Inputs(
        announcement_matrix=announcement_matrix,
        pre_auction_matrix=pre_auction_matrix,
        target_table=target_table,
        announcement_audit=announcement_audit,
        pre_auction_audit=pre_auction_audit,
        pre_auction_update_table=pre_auction_update_table,
        fingerprints=fingerprints,
    )


def validate_phase6_inputs(inputs: Phase6Inputs, *, contract_path: Path = DEFAULT_CONTRACT_PATH) -> dict:
    """Every check `docs/project_rules.md`/the Phase 6 task specification requires
    before any modeling touches these artifacts. Raises on the first
    violation found; returns a small results dict (row counts, gate
    results) for the protocol report on success.
    """
    contract = load_contract(contract_path)
    entries = parse_entries(contract)

    n_rows = len(inputs.announcement_matrix)
    validate_predictor_matrix(inputs.announcement_matrix, entries, expected_rows=n_rows)
    validate_predictor_matrix(inputs.pre_auction_matrix, entries, expected_rows=n_rows)
    validate_target_table(inputs.target_table, entries, expected_rows=n_rows)
    validate_audit_table(inputs.announcement_audit, inputs.announcement_matrix, cutoff_name="announcement")
    validate_audit_table(inputs.pre_auction_audit, inputs.pre_auction_matrix, cutoff_name="pre_auction")
    validate_pre_auction_update_table(inputs.pre_auction_update_table, inputs.announcement_matrix)

    # Identical evaluation key sets between the two cutoff views.
    ann_keys = set(inputs.announcement_matrix[AUCTION_KEY_COL])
    pre_keys = set(inputs.pre_auction_matrix[AUCTION_KEY_COL])
    target_keys = set(inputs.target_table[AUCTION_KEY_COL])
    if not (ann_keys == pre_keys == target_keys):
        raise AssertionError(
            "Phase 6: announcement matrix, pre-auction matrix, and target table key sets disagree"
        )

    # No missing primary target among evaluated settled auctions.
    if inputs.target_table["primary_dealer_share"].isna().any():
        bad = inputs.target_table.loc[
            inputs.target_table["primary_dealer_share"].isna(), AUCTION_KEY_COL
        ].tolist()
        raise AssertionError(f"Phase 6: primary_dealer_share missing for evaluated auctions: {bad[:10]}")

    # No result/target column inside predictor inputs.
    for name, matrix in (("announcement", inputs.announcement_matrix), ("pre_auction", inputs.pre_auction_matrix)):
        scan = forbidden_field_scan(list(matrix.columns))
        if scan["exact_matches"] or scan["pattern_matches"]:
            raise AssertionError(f"Phase 6: {name} matrix contains forbidden result/target-like columns: {scan}")

    # All selected source safe-availability dates at or before cutoff.
    availability_results = {
        "announcement": assert_audit_table_availability_safe(
            inputs.announcement_audit, inputs.announcement_matrix, cutoff_col=CUTOFF_COL_BY_NAME["announcement"]
        ),
        "pre_auction": assert_audit_table_availability_safe(
            inputs.pre_auction_audit, inputs.pre_auction_matrix, cutoff_col=CUTOFF_COL_BY_NAME["pre_auction"]
        ),
    }

    return {
        "n_rows": n_rows,
        "fingerprints": inputs.fingerprints,
        "availability_results": availability_results,
    }
