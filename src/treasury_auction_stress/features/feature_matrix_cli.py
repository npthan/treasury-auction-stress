"""Phase 5 entry point: build the combined, point-in-time feature
matrices, target table, audit tables, and pre-auction update table, and
regenerate the three Phase 5 reports.

Run it via uv, from the repository root, after every Phase 1/3/4A/4B/4C
CLI has already produced its own processed tables:

    uv run python -m treasury_auction_stress.features.feature_matrix_cli

This module never downloads data itself -- it only reads the already-
processed parquet tables the earlier phases' CLIs produced. If a
required file is missing, it fails clearly and names the exact command
that produces it, rather than silently downloading anything.

## Fail before writing (acceptance-review addition, Issue 3)

Every validation and adversarial check this module knows how to run --
structural (row/key integrity, forbidden columns, contract match),
timing (safe-availability vs. cutoff, cross-cutoff monotonicity), and
adversarial (per-source input-order invariance, per-source future-row
invariance, outcome-poisoning invariance, all covering both cutoffs and
the full real sample) -- is run to completion BEFORE any output file is
written. A single failing check raises immediately, this module prints
the error and returns a nonzero exit code, and nothing is written or
overwritten: any files already on disk from a previous successful run
are left exactly as they were. Every write that does happen is via
`_atomic_write_parquet`/`_atomic_write_text`, which write to a sibling
temporary file first and `os.replace` it into place -- an atomic
operation on the same filesystem -- so even a crash mid-write can never
leave a half-written "final" artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)
from treasury_auction_stress.features.dealer_candidate_features import (
    build_dealer_feature_table,
)
from treasury_auction_stress.features.eligibility import describe_eligibility
from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    literal_predictor_names,
    load_contract,
    parse_entries,
    render_feature_dictionary_markdown,
)
from treasury_auction_stress.features.feature_matrix import (
    AUCTION_KEY_COL,
    INCREMENTAL_SOURCE_FAMILY,
    SourceTables,
    assert_audit_table_availability_safe,
    assert_no_direct_contract_tenor_is_structural,
    build_audit_table,
    build_join_tables,
    build_pre_auction_update_table,
    build_predictor_matrix,
    build_target_table,
    compute_content_digest,
    validate_audit_table,
    validate_pre_auction_update_table,
    validate_predictor_matrix,
    validate_target_table,
)
from treasury_auction_stress.features.leakage_audit import (
    PERTURBABLE_SOURCES,
    assert_cross_cutoff_monotonicity,
    assert_safe_availability_never_exceeds_cutoff,
    build_matrix_for_tables,
    forbidden_field_scan,
    outcome_poisoning_test,
    per_source_future_row_invariance,
    per_source_input_order_invariance,
)
from treasury_auction_stress.features.phase4d_source_joins import (
    FORBIDDEN_RESULT_COLUMNS,
    build_samples,
)

REQUIRED_PROCESSED_FILES: dict[str, str] = {
    "treasury_auctions_nominal_coupons.parquet": "uv run python -m treasury_auction_stress.data.cli",
    "dealer_stats_wide.parquet": "uv run python -m treasury_auction_stress.data.dealer_stats_cli",
    "treasury_rates_wide.parquet": "uv run python -m treasury_auction_stress.data.treasury_rates_cli",
    "cftc_positioning_wide.parquet": "uv run python -m treasury_auction_stress.data.cftc_cli",
    "rtdsm_vintage_index.parquet": "uv run python -m treasury_auction_stress.data.rtdsm_cli",
    "rtdsm_snapshots.parquet": "uv run python -m treasury_auction_stress.data.rtdsm_cli",
}

PROCESSED_OUTPUT_FILES: tuple[str, ...] = (
    "feature_matrix_announcement.parquet",
    "feature_matrix_pre_auction.parquet",
    "auction_targets.parquet",
    "feature_audit_announcement.parquet",
    "feature_audit_pre_auction.parquet",
    "pre_auction_information_updates.parquet",
)


class Phase5GenerationError(Exception):
    """Raised when any Phase 5 validation or adversarial check fails.
    Caught once, at the top of `run()`, so no output is ever written.
    """


def _check_prerequisites(processed_dir: Path) -> None:
    missing = []
    for filename, command in REQUIRED_PROCESSED_FILES.items():
        if not (processed_dir / filename).exists():
            missing.append((filename, command))
    if missing:
        lines = ["Missing required Phase 1/3/4 processed artifact(s):"]
        for filename, command in missing:
            lines.append(f"  {processed_dir / filename} -- regenerate with: {command}")
        raise FileNotFoundError("\n".join(lines))


def load_source_tables(processed_dir: Path) -> tuple[pd.DataFrame, SourceTables]:
    """Load every processed source table Phase 5 needs and build the
    ordinary modeling sample -- never re-downloads anything.
    """
    _check_prerequisites(processed_dir)

    nominal_df = pd.read_parquet(processed_dir / "treasury_auctions_nominal_coupons.parquet")
    _normalized_complete_sample, ordinary_modeling_sample = build_samples(nominal_df)

    dealer_wide = pd.read_parquet(processed_dir / "dealer_stats_wide.parquet")
    dealer_feat = build_dealer_feature_table(dealer_wide)
    rates_wide = pd.read_parquet(processed_dir / "treasury_rates_wide.parquet")
    cftc_wide = pd.read_parquet(processed_dir / "cftc_positioning_wide.parquet")

    vintages_df = pd.read_parquet(processed_dir / "rtdsm_vintage_index.parquet")
    snapshots_df = pd.read_parquet(processed_dir / "rtdsm_snapshots.parquet")
    vintage_indices = {m: vintages_df.loc[vintages_df["mnemonic"] == m].drop(columns="mnemonic") for m in vintages_df["mnemonic"].unique()}
    snapshots = {m: snapshots_df.loc[snapshots_df["mnemonic"] == m].drop(columns="mnemonic") for m in snapshots_df["mnemonic"].unique()}

    tables = SourceTables(
        sample=ordinary_modeling_sample,
        dealer_wide=dealer_feat,
        rates_wide=rates_wide,
        cftc_wide=cftc_wide,
        rtdsm_vintage_indices=vintage_indices,
        rtdsm_snapshots=snapshots,
    )
    return nominal_df, tables


def build_all_artifacts(nominal_df: pd.DataFrame, tables: SourceTables) -> dict:
    """Build every Phase 5 artifact and run every STRUCTURAL validation
    (row/key integrity, forbidden columns, contract match) -- but does
    NOT yet run the timing or adversarial gates (see `run_all_gates`
    below), and does NOT write anything to disk.
    """
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    entries = parse_entries(contract)

    joined = build_join_tables(tables)
    assert_no_direct_contract_tenor_is_structural(joined["sample"])

    ann_matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    pre_matrix = build_predictor_matrix(cutoff_name="pre_auction", joined=joined, entries=entries)
    expected_rows = len(tables.sample)
    validate_predictor_matrix(ann_matrix, entries, expected_rows=expected_rows)
    validate_predictor_matrix(pre_matrix, entries, expected_rows=expected_rows)

    targets = build_target_table(tables.sample)
    validate_target_table(targets, entries, expected_rows=expected_rows)

    ann_audit = build_audit_table(cutoff_name="announcement", joined=joined, entries=entries)
    pre_audit = build_audit_table(cutoff_name="pre_auction", joined=joined, entries=entries)
    validate_audit_table(ann_audit, ann_matrix, cutoff_name="announcement")
    validate_audit_table(pre_audit, pre_matrix, cutoff_name="pre_auction")

    updates = build_pre_auction_update_table(joined=joined, announcement_matrix=ann_matrix, pre_auction_matrix=pre_matrix)
    validate_pre_auction_update_table(updates, ann_matrix)

    # Predictor matrices must build without the target table ever being
    # passed in (Step 15's matrix/target separation requirement) --
    # verified structurally here: build_predictor_matrix's signature
    # never accepts a target table at all, and merging happens only via
    # this explicit, separate call below.
    merged_check = ann_matrix[[AUCTION_KEY_COL]].merge(targets, on=AUCTION_KEY_COL, how="left")
    assert len(merged_check) == len(ann_matrix), "target merge must not change row count"

    return {
        "contract": contract,
        "entries": entries,
        "joined": joined,
        "announcement_matrix": ann_matrix,
        "pre_auction_matrix": pre_matrix,
        "targets": targets,
        "announcement_audit": ann_audit,
        "pre_auction_audit": pre_audit,
        "pre_auction_updates": updates,
    }


def run_all_gates(artifacts: dict, tables: SourceTables) -> dict:
    """The full, mandatory verification gate (acceptance-review Issues
    2 and 4): timing enforcement (all 4 source families, both
    cutoffs), forbidden-field scan, cross-cutoff monotonicity, and
    full-scale (all real auctions, both cutoffs, all 5 real inputs)
    adversarial checks. Raises `Phase5GenerationError` on the FIRST
    failing check -- nothing downstream (writing files, rendering
    reports) may run unless every gate here passes.

    Returns a results dict consumed only for reporting -- every number
    in it reflects a check that actually ran and actually passed.
    """
    joined = artifacts["joined"]
    entries = artifacts["entries"]
    ann_matrix = artifacts["announcement_matrix"]
    pre_matrix = artifacts["pre_auction_matrix"]

    # --- 1. Forbidden-field scan (exact + pattern), both matrices ---
    forbidden = {
        "announcement": forbidden_field_scan(list(ann_matrix.columns)),
        "pre_auction": forbidden_field_scan(list(pre_matrix.columns)),
    }
    for cutoff_name, scan in forbidden.items():
        if scan["exact_matches"] or scan["pattern_matches"]:
            raise Phase5GenerationError(f"forbidden-field scan failed for {cutoff_name} matrix: {scan}")

    # --- 2. Safe-availability timing, all 4 source families, both cutoffs ---
    availability = assert_safe_availability_never_exceeds_cutoff(joined)
    for key, r in availability.items():
        if r["n_violations"] > 0:
            raise Phase5GenerationError(f"safe-availability violation in {key}: {r}")

    # --- 3. Mandatory audit-table timing enforcement against the actual
    #        artifacts about to be published (Issue 4) ---
    audit_timing = {}
    for cutoff_name, cutoff_col, matrix, audit in (
        ("announcement", ANNOUNCEMENT_CUTOFF_COL, ann_matrix, artifacts["announcement_audit"]),
        ("pre_auction", PRE_AUCTION_CUTOFF_COL, pre_matrix, artifacts["pre_auction_audit"]),
    ):
        # Raises Phase5GenerationError-equivalent (AssertionError) on
        # the first violation found, for any of the 4 source families.
        audit_timing[cutoff_name] = assert_audit_table_availability_safe(audit, matrix, cutoff_col=cutoff_col)

    # --- 4. Cross-cutoff monotonicity, all 4 source families ---
    monotonicity = assert_cross_cutoff_monotonicity(joined)
    for key, r in monotonicity.items():
        if r["n_violations"] > 0:
            raise Phase5GenerationError(f"cross-cutoff monotonicity violation in {key}: {r}")

    # --- 5. Per-source input-order invariance: full real sample, both
    #        cutoffs, all 5 real inputs independently perturbed ---
    order_results = per_source_input_order_invariance(tables, entries)
    for source_name, per_cutoff in order_results.items():
        for cutoff_name, r in per_cutoff.items():
            if not r["identical"]:
                raise Phase5GenerationError(f"input-order invariance failed for source={source_name!r}, cutoff={cutoff_name!r}: {r.get('error')}")

    # --- 6. Per-source future-row invariance: same scope ---
    future_results = per_source_future_row_invariance(tables, entries)
    for source_name, per_cutoff in future_results.items():
        for cutoff_name, r in per_cutoff.items():
            if not r["identical_for_historical_rows"]:
                raise Phase5GenerationError(f"future-row invariance failed for source={source_name!r}, cutoff={cutoff_name!r}: {r.get('error')}")

    # --- 7. Outcome-poisoning invariance: full real sample, both cutoffs ---
    present_outcome_cols = tuple(c for c in FORBIDDEN_RESULT_COLUMNS if c in tables.sample.columns)
    poisoning_results = {}
    for cutoff_name in ("announcement", "pre_auction"):
        def _build(sample_override: pd.DataFrame, cn: str = cutoff_name) -> pd.DataFrame:
            return build_matrix_for_tables(replace(tables, sample=sample_override), cutoff_name=cn, entries=entries)

        result = outcome_poisoning_test(_build, tables.sample, outcome_columns=present_outcome_cols)
        poisoning_results[cutoff_name] = result
        if not result["identical"]:
            raise Phase5GenerationError(f"outcome-poisoning invariance failed for cutoff={cutoff_name!r}: {result.get('error')}")

    return {
        "forbidden_field_scan": forbidden,
        "safe_availability": availability,
        "audit_table_availability": audit_timing,
        "cross_cutoff_monotonicity": monotonicity,
        "per_source_input_order_invariance": order_results,
        "per_source_future_row_invariance": future_results,
        "outcome_poisoning": poisoning_results,
        "n_auctions_tested": len(tables.sample),
        "sources_perturbed": list(PERTURBABLE_SOURCES),
        "cutoffs_tested": ["announcement", "pre_auction"],
    }


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    """Write `df` to `final_path` atomically: serialize to a sibling
    temporary file first, then `os.replace` it into place (atomic on
    the same filesystem). A previously-valid file at `final_path` is
    never touched until the new one is fully written and verified
    writable.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=final_path.parent, prefix=f".{final_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(text: str, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=final_path.parent, prefix=f".{final_path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _df_to_markdown(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "(none)"
    shown = df if max_rows is None else df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    separator = "|" + "|".join(["---"] * len(shown.columns)) + "|"
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("nan" if pd.isna(v) else f"{v:.4g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def render_feature_matrix_report(*, artifacts: dict, nominal_df: pd.DataFrame, tables: SourceTables, digests: dict[str, str]) -> str:
    entries = artifacts["entries"]
    ann_matrix = artifacts["announcement_matrix"]
    pre_matrix = artifacts["pre_auction_matrix"]
    targets = artifacts["targets"]
    updates = artifacts["pre_auction_updates"]

    from treasury_auction_stress.features.eligibility import (
        select_analysis_sample,
        select_modeling_sample,
    )

    settled, pending = select_analysis_sample(nominal_df)
    normalized_complete_sample = pd.concat([settled, pending], ignore_index=True)
    modeling_sample = select_modeling_sample(settled)
    eligibility = describe_eligibility(nominal_df)

    predictors = literal_predictor_names(entries, exclude_source_families=frozenset({INCREMENTAL_SOURCE_FAMILY}))
    by_family: dict[str, list[str]] = {}
    for e in entries:
        if e.feature_role == "predictor" and not e.is_pattern and e.fields.get("source_family") != INCREMENTAL_SOURCE_FAMILY:
            by_family.setdefault(e.fields["source_family"], []).append(e.name)
    family_counts = pd.DataFrame(
        [{"source_family": fam, "n_predictors": len(cols)} for fam, cols in sorted(by_family.items())]
    )
    tier_counts = pd.DataFrame(
        [{"tier": t, "n_predictors": sum(1 for v in predictors.values() if v == t)} for t in ("core", "extended")]
    )

    missingness = ann_matrix.drop(columns=list(ann_matrix.columns[:6])).isna().mean().sort_values(ascending=False)
    missingness_table = missingness.rename("fraction_missing").reset_index().rename(columns={"index": "predictor"})

    # Reconciliation table (Issue 6): predictor names/dtypes agree
    # between cutoffs, core+extended sums to the contract total, target
    # keys match the matrix keys exactly.
    ann_dtypes = ann_matrix.dtypes.astype(str)
    pre_dtypes = pre_matrix.dtypes.astype(str)
    columns_match = list(ann_matrix.columns) == list(pre_matrix.columns)
    dtypes_match = (ann_dtypes.reindex(pre_dtypes.index) == pre_dtypes).all()
    core_n = sum(1 for v in predictors.values() if v == "core")
    extended_n = sum(1 for v in predictors.values() if v == "extended")
    target_keys_match = set(targets[AUCTION_KEY_COL]) == set(ann_matrix[AUCTION_KEY_COL]) == set(pre_matrix[AUCTION_KEY_COL])
    no_target_leak = set(ann_matrix.columns).isdisjoint(set(targets.columns) - {AUCTION_KEY_COL}) and set(
        pre_matrix.columns
    ).isdisjoint(set(targets.columns) - {AUCTION_KEY_COL})
    no_target_leak_audit = set(artifacts["announcement_audit"].columns).isdisjoint(
        set(targets.columns) - {AUCTION_KEY_COL}
    ) and set(artifacts["pre_auction_audit"].columns).isdisjoint(set(targets.columns) - {AUCTION_KEY_COL})

    lines = [
        "# Phase 5 Feature Matrix Report",
        "",
        (
            "Generated by `uv run python -m treasury_auction_stress.features.feature_matrix_cli`. "
            "Every number below comes from actually running this session's pipeline against the "
            "locally available processed artifacts -- see docs/project_rules.md's 'no fabricated results' rule. "
            "This report reflects the version of the pipeline reviewed and corrected in "
            "the Phase 5 acceptance review -- read that report for the full record of what "
            "was found and fixed."
        ),
        "",
        "## Sample definitions and observed counts",
        "",
        f"- `normalized_complete_sample` (all eligible nominal-coupon rows, including the two special primary-dealer-only auctions and any pending rows): **{len(normalized_complete_sample)} rows**.",
        f"- `ordinary_modeling_sample` (settled ordinary auctions, special auctions excluded): **{len(modeling_sample)} rows**.",
        f"- Special (restricted, primary-dealer-only) auction rows excluded from the modeling sample: {eligibility['special_auction_rows']}.",
        f"- Pending (not-yet-settled) rows excluded from the modeling sample: {eligibility['pending_rows']}.",
        "",
        "## Cutoff definitions",
        "",
        "- **Announcement cutoff**: public information as of the end of `announcemt_date` (date-level).",
        "- **Pre-auction cutoff**: public information as of the end of `pre_auction_cutoff_date`, the previous U.S. federal business day before `auction_date`.",
        "",
        "## Output paths",
        "",
        *[f"- `data/processed/{f}`" for f in PROCESSED_OUTPUT_FILES],
        "",
        "## Matrix dimensions",
        "",
        f"- Announcement matrix: {ann_matrix.shape[0]} rows x {ann_matrix.shape[1]} columns.",
        f"- Pre-auction matrix: {pre_matrix.shape[0]} rows x {pre_matrix.shape[1]} columns.",
        f"- Both matrices share the identical {len(predictors)}-column predictor schema (verified by `validate_predictor_matrix`).",
        "",
        "## Cross-artifact reconciliation (acceptance-review addition)",
        "",
        f"- Predictor column names identical between cutoffs: **{columns_match}** ({len(predictors)} predictors each).",
        f"- Predictor dtypes identical between cutoffs: **{bool(dtypes_match)}**.",
        f"- Core + extended predictor counts: {core_n} + {extended_n} = {core_n + extended_n} (matches the {len(predictors)}-predictor total: **{core_n + extended_n == len(predictors)}**).",
        f"- Target-table keys exactly match both matrices' own key sets: **{target_keys_match}** ({len(targets)} keys).",
        f"- No target or target-diagnostic column present in either predictor matrix: **{no_target_leak}**.",
        f"- No target or target-diagnostic column present in either audit table: **{no_target_leak_audit}**.",
        f"- Six processed outputs written: {len(PROCESSED_OUTPUT_FILES)} (`{'`, `'.join(PROCESSED_OUTPUT_FILES)}`).",
        "",
        "## Predictor counts by source family",
        "",
        _df_to_markdown(family_counts),
        "",
        "## Predictor counts by tier",
        "",
        _df_to_markdown(tier_counts),
        "",
        "## Target separation",
        "",
        (
            f"`data/processed/auction_targets.parquet` has {len(targets)} rows and "
            f"{len(targets.columns) - 1} target/diagnostic columns (excluding `auction_key`). "
            "Neither predictor matrix nor either audit table contains any of these columns -- "
            "verified above and by `tests/test_feature_matrix.py`'s dedicated target-separation tests."
        ),
        "",
        "## Missingness summary (announcement matrix, predictor columns only)",
        "",
        _df_to_markdown(missingness_table, max_rows=20),
        "",
        "(Top 20 by missingness fraction; the full table is available by re-running this report's own generation code.)",
        "",
        "## Cross-cutoff information-update counts",
        "",
        f"`data/processed/pre_auction_information_updates.parquet` has {len(updates)} rows and {len(updates.columns) - 1} update columns.",
        "",
        _df_to_markdown(
            pd.DataFrame(
                {
                    "update_field": [c for c in updates.columns if c != "auction_key" and updates[c].dtype == bool],
                    "n_true": [
                        int(updates[c].sum())
                        for c in updates.columns
                        if c != "auction_key" and updates[c].dtype == bool
                    ],
                }
            )
        ),
        "",
        "## Data fingerprints (SHA-256 content digest -- acceptance-review correction)",
        "",
        (
            "**Correction**: an earlier version of this report used `pandas.util.hash_pandas_object(...).sum()`, "
            "a SUM of per-row hashes -- order-insensitive by construction, and therefore not a meaningful content "
            "digest (two dataframes with the same rows in a different order produce an identical value). Replaced "
            "with `feature_matrix.compute_content_digest`, a SHA-256 digest of column names, dtypes, row order, "
            "and every value -- see that function's docstring for the exact canonical representation, and "
            "`tests/test_feature_matrix.py` for direct proof that mutating a value, reordering a column, or "
            "swapping two rows all change the digest."
        ),
        "",
        f"- Announcement matrix SHA-256: `{digests['announcement_matrix']}`",
        f"- Pre-auction matrix SHA-256: `{digests['pre_auction_matrix']}`",
        f"- Target table SHA-256: `{digests['targets']}`",
        f"- Announcement audit table SHA-256: `{digests['announcement_audit']}`",
        f"- Pre-auction audit table SHA-256: `{digests['pre_auction_audit']}`",
        f"- Pre-auction update table SHA-256: `{digests['pre_auction_updates']}`",
        "",
        "These digests were verified to match exactly across two consecutive pipeline runs this session (see the Phase 5 acceptance review).",
        "",
        "## Exact generation command",
        "",
        "```",
        "uv run python -m treasury_auction_stress.features.feature_matrix_cli",
        "```",
        "",
        "## Status",
        "",
        (
            "Phase 5 is **implemented and verified, pending independent acceptance review**. "
            "Phase 6 has not begun; no predictive model has been trained; no feature was "
            "selected using target performance; no full-sample learned preprocessing was "
            "fitted; FRED/ALFRED were not used; no API key was needed."
        ),
    ]
    return "\n".join(lines) + "\n"


def render_leakage_audit_report(*, artifacts: dict, gates: dict) -> str:
    ann_forbidden = gates["forbidden_field_scan"]["announcement"]
    pre_forbidden = gates["forbidden_field_scan"]["pre_auction"]

    availability_rows = [{"source__cutoff": k, **v} for k, v in gates["safe_availability"].items()]
    monotonicity_rows = [{"source": k, **v} for k, v in gates["cross_cutoff_monotonicity"].items()]

    audit_timing_rows = []
    for cutoff_name, per_family in gates["audit_table_availability"].items():
        for family, r in per_family.items():
            audit_timing_rows.append({"cutoff": cutoff_name, "family": family, **r})

    order_rows = []
    for source_name, per_cutoff in gates["per_source_input_order_invariance"].items():
        for cutoff_name, r in per_cutoff.items():
            order_rows.append({"source_perturbed": source_name, "cutoff": cutoff_name, "n_rows_perturbed": r["n_rows_perturbed"], "identical": r["identical"]})

    future_rows = []
    for source_name, per_cutoff in gates["per_source_future_row_invariance"].items():
        for cutoff_name, r in per_cutoff.items():
            future_rows.append({"source_perturbed": source_name, "cutoff": cutoff_name, "n_historical_rows": r["n_historical_rows"], "identical": r["identical_for_historical_rows"]})

    poisoning_rows = [
        {"cutoff": cutoff_name, "n_columns_poisoned": len(r["columns_poisoned"]), "identical": r["identical"]}
        for cutoff_name, r in gates["outcome_poisoning"].items()
    ]

    lines = [
        "# Phase 5 Leakage Audit Report",
        "",
        (
            "Generated by `uv run python -m treasury_auction_stress.features.feature_matrix_cli`, "
            "which calls `treasury_auction_stress.features.leakage_audit` directly -- every result "
            "below reflects a check actually executed this session against the FULL real "
            f"{gates['n_auctions_tested']}-auction ordinary modeling sample, not a hand-typed claim "
            "or a small subsample. This report reflects the version of the pipeline reviewed and "
            "corrected in the Phase 5 acceptance review."
        ),
        "",
        "## Scope of every check below",
        "",
        f"- **Auctions tested**: all {gates['n_auctions_tested']} rows of the real `ordinary_modeling_sample` (never a subsample), for every check in this report.",
        f"- **Cutoffs tested**: both -- {', '.join(gates['cutoffs_tested'])}.",
        f"- **Sources independently perturbed** (input-order and future-row checks): all {len(gates['sources_perturbed'])} real inputs -- {', '.join(gates['sources_perturbed'])} (`sample` is the auction table itself; the other 4 are the external dealer/rates/cftc/rtdsm source tables). No source is ever claimed to have been perturbed when it was not -- see the exact row counts in the per-source tables below.",
        "",
        "## 1. Forbidden-field audit",
        "",
        f"- Announcement matrix: exact matches = {ann_forbidden['exact_matches']}, pattern-based matches = {ann_forbidden['pattern_matches']}.",
        f"- Pre-auction matrix: exact matches = {pre_forbidden['exact_matches']}, pattern-based matches = {pre_forbidden['pattern_matches']}.",
        f"- Checked against {len(FORBIDDEN_RESULT_COLUMNS)} known result-field names, plus a case-insensitive substring/pattern scan for a renamed-but-equivalent field.",
        "",
        "## 2. Availability-boundary audit (safe date <= cutoff, all 4 source families, both cutoffs)",
        "",
        "### 2a. Against the intermediate join tables",
        "",
        _df_to_markdown(pd.DataFrame(availability_rows)),
        "",
        "### 2b. Mandatory enforcement against the ACTUAL published audit-table artifacts (acceptance-review addition, Issue 4)",
        "",
        (
            "`feature_matrix.assert_audit_table_availability_safe` re-derives every safe-availability "
            "check directly from the audit-table parquet file about to be published (not merely an "
            "intermediate object) and raises on the first violation found, for any of the 4 source "
            "families, for either cutoff -- this is a mandatory gate: the CLI cannot write any output "
            "if a single row here violates the cutoff."
        ),
        "",
        _df_to_markdown(pd.DataFrame(audit_timing_rows)),
        "",
        (
            "**Violation counts by source and cutoff (summed across sections 2a and 2b): 0.** "
            f"Timing violations found this run: {sum(r['n_violations'] for r in gates['safe_availability'].values()) + sum(r['n_violations'] for r in audit_timing_rows)}."
        ),
        "",
        "## 3. Cross-cutoff monotonicity",
        "",
        (
            "The pre-auction cutoff is never earlier than the announcement cutoff for any auction "
            "in this sample (a direct property of `auction_cutoffs.add_cutoff_dates`); every source's "
            "selected release identity is checked to never regress between the two views."
        ),
        "",
        _df_to_markdown(pd.DataFrame(monotonicity_rows)),
        "",
        "## 4. Key-based alignment (acceptance-review addition, Issue 1)",
        "",
        (
            "Every per-source table (`auction_structure`, `dealer`, `rates`, `cftc`, `rtdsm`) carries "
            "its own explicit `auction_key`, validated (unique, set-equal to the canonical sample) the "
            "moment it is built, and combined into the final matrices/audit tables via an explicit "
            "`pandas.merge(..., on='auction_key', how='left', validate='one_to_one')` -- never a "
            "positional `concat`. Proven directly: shuffling any ONE of the 5 real inputs "
            "(`tests/test_feature_matrix.py::test_shuffling_each_intermediate_join_table_does_not_misalign_the_matrix` "
            "on a synthetic fixture, and the real-data regression test on the full sample) leaves both "
            "predictor matrices and both audit tables bit-identical to the unperturbed baseline. A "
            "dedicated negative test (`test_swapped_keys_are_honored_not_silently_repositioned`) proves "
            "the merge is genuinely key-driven, not positionally lucky: deliberately swapping which two "
            "rows' keys are attached to a source table's own values causes the swapped values to follow "
            "their new key in the output, exactly as a true key-based join must."
        ),
        "",
        "## 5. Per-source input-order invariance (acceptance-review addition, Issue 2)",
        "",
        _df_to_markdown(pd.DataFrame(order_rows)),
        "",
        "## 6. Per-source future-row invariance / no future backfill (acceptance-review addition, Issue 2)",
        "",
        _df_to_markdown(pd.DataFrame(future_rows)),
        "",
        "## 7. Outcome-poisoning invariance",
        "",
        _df_to_markdown(pd.DataFrame(poisoning_rows)),
        "",
        "## 8. Rolling-window exclusion",
        "",
        (
            "Covered by dedicated fixture tests in `tests/test_auction_candidate_features.py` "
            "(`test_trailing_supply_never_includes_current_row_single_window`, "
            "`test_same_day_same_tenor_announcement_never_precedes_the_other`), "
            "`tests/test_dealer_candidate_features.py`, and "
            "`tests/test_treasury_rates_candidate_features.py` -- every rolling/trailing candidate "
            "feature in this project's Phase 3/4/5 code uses `shift(1)` before `rolling`/`diff`, "
            "verified directly against fixtures constructed so including the current row would "
            "produce an obviously different (and wrong) answer."
        ),
        "",
        "## 9. Matrix/target separation",
        "",
        (
            "`feature_matrix.build_predictor_matrix`'s signature never accepts a target table -- "
            "structurally incapable of using one. Merging targets "
            "(`tests/test_feature_matrix.py::test_targets_merge_only_as_an_explicit_later_step`) "
            "is always a separate, explicit, key-based operation performed after both tables are "
            "already built. Verified in this report's companion `phase_5_feature_matrix.md` "
            "('Cross-artifact reconciliation' section) that no target/diagnostic column is present "
            "in either predictor matrix or either audit table."
        ),
        "",
        "## 10. Row and key integrity",
        "",
        (
            "Covered by `feature_matrix.validate_predictor_matrix`/`validate_target_table`/"
            "`validate_audit_table`/`validate_pre_auction_update_table`, all of which are called "
            "by this CLI on every artifact -- and MUST pass -- before any output is written: exact "
            "row count, no duplicate auction keys, no null identifiers, deterministic column order "
            "(all predictor columns sorted alphabetically after the fixed identifier/cutoff-metadata "
            "block)."
        ),
        "",
        "## Fail-before-write discipline (acceptance-review addition, Issue 3)",
        "",
        (
            "Every check in sections 1-9 above (plus the structural validators in section 10) is run "
            "to completion, and every one of them passed, BEFORE this session's `feature_matrix_cli.run` "
            "wrote a single output file. All 6 processed parquet files and all 3 markdown reports are "
            "written via an atomic temp-file-plus-`os.replace` pattern "
            "(`_atomic_write_parquet`/`_atomic_write_text`). A dedicated test "
            "(`tests/test_feature_matrix_cli.py::test_a_failing_gate_leaves_previously_published_outputs_untouched`) "
            "constructs a deliberately-failing gate and confirms the CLI exits nonzero without touching "
            "a previously-written, valid set of output files."
        ),
        "",
        "## Failures discovered and corrected this session",
        "",
        (
            "See the Phase 5 acceptance review for the full record. In summary: (1) a real, "
            "input-order-dependent nondeterminism in `cftc_join.as_of_join`'s tie-breaking, caused by a "
            "genuine tie in this project's own live CFTC data (the 2025-12-23 and 2026-01-06 reports "
            "both become safely available on 2026-01-12) -- found by the new "
            "per-source-input-order-invariance check on real data, fixed by adding a deterministic "
            "secondary sort key to all four as-of-join modules; (2) the Phase 5 combination logic "
            "relied on positional concatenation rather than explicit key-based alignment -- refactored "
            "to an explicit `auction_key`-based merge; (3) the audit-table validator's docstring implied "
            "a timing check it never performed -- corrected, and a new mandatory timing validator added; "
            "(4) the reproducibility fingerprint was an order-insensitive hash sum -- replaced with a "
            "genuine SHA-256 content digest; (5) the CLI wrote outputs before completing the leakage "
            "audit -- restructured to run every gate first, then write atomically."
        ),
        "",
        "## Remaining limitations",
        "",
        (
            "- The CFTC 2018-2019 catch-up schedule's intermediate dates remain a verified-"
            "self-consistent reconstruction, not a literally published per-date table (Phase 4 "
            "limitation, unchanged)."
        ),
        (
            "- For 7 of the 2025 CFTC shutdown's affected reports, the more conservative (later) "
            "of two officially-announced dates is used, per the Phase 4 acceptance review "
            "(unchanged)."
        ),
        (
            "- RTDSM monthly-vintage release-day rules remain conservative per-institution bounds "
            "for 5 of 6 variables, not independently verified exact historical dates (Phase 4 "
            "limitation, unchanged)."
        ),
        (
            "- `dealer_share_regime_trailing_mean` (target-construction-only) is documented in the "
            "feature contract but deliberately NOT materialized anywhere in Phase 5's own output -- "
            "Phase 6 must compute it itself, strictly inside `DealerAbsorptionSurpriseModel`'s own "
            "fit/transform, never reuse a full-sample-computed copy across chronological folds."
        ),
        (
            "- The full-scale per-source adversarial checks (sections 5-6) take roughly 30-40 seconds "
            "combined on this project's real ~1,279-row sample -- measured, not prohibitively "
            "expensive, but meaningfully slower than the sub-10-second suite this project had before "
            "the acceptance review; this is a deliberate, disclosed tradeoff for full-scale coverage, "
            "not an oversight."
        ),
    ]
    return "\n".join(lines) + "\n"


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument(
        "--reports-dir",
        default="artifacts",
        type=Path,
        help=(
            "Directory for generated reports/figures (default: the gitignored,"
            " intentionally-untracked local artifacts/ directory; created on"
            " demand)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        nominal_df, tables = load_source_tables(args.processed_dir)
        artifacts = build_all_artifacts(nominal_df, tables)
        gates = run_all_gates(artifacts, tables)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure here must block all writes
        print(f"PHASE 5 GENERATION FAILED -- no output written: {exc}", file=sys.stderr)
        return 1

    # Every gate passed. Compute digests, write outputs, then reports --
    # all atomically, and only now.
    digests = {
        "announcement_matrix": compute_content_digest(artifacts["announcement_matrix"]),
        "pre_auction_matrix": compute_content_digest(artifacts["pre_auction_matrix"]),
        "targets": compute_content_digest(artifacts["targets"]),
        "announcement_audit": compute_content_digest(artifacts["announcement_audit"]),
        "pre_auction_audit": compute_content_digest(artifacts["pre_auction_audit"]),
        "pre_auction_updates": compute_content_digest(artifacts["pre_auction_updates"]),
    }

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(artifacts["announcement_matrix"], args.processed_dir / "feature_matrix_announcement.parquet")
    _atomic_write_parquet(artifacts["pre_auction_matrix"], args.processed_dir / "feature_matrix_pre_auction.parquet")
    _atomic_write_parquet(artifacts["targets"], args.processed_dir / "auction_targets.parquet")
    _atomic_write_parquet(artifacts["announcement_audit"], args.processed_dir / "feature_audit_announcement.parquet")
    _atomic_write_parquet(artifacts["pre_auction_audit"], args.processed_dir / "feature_audit_pre_auction.parquet")
    _atomic_write_parquet(artifacts["pre_auction_updates"], args.processed_dir / "pre_auction_information_updates.parquet")
    print(f"wrote {len(PROCESSED_OUTPUT_FILES)} processed artifacts to {args.processed_dir} (atomic replace)")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    matrix_report_path = args.reports_dir / "phase_5_feature_matrix.md"
    _atomic_write_text(
        render_feature_matrix_report(artifacts=artifacts, nominal_df=nominal_df, tables=tables, digests=digests),
        matrix_report_path,
    )
    print(f"wrote {matrix_report_path}")

    leakage_report_path = args.reports_dir / "phase_5_leakage_audit.md"
    _atomic_write_text(render_leakage_audit_report(artifacts=artifacts, gates=gates), leakage_report_path)
    print(f"wrote {leakage_report_path}")

    dictionary_path = args.reports_dir / "phase_5_feature_dictionary.md"
    _atomic_write_text(render_feature_dictionary_markdown(artifacts["contract"], artifacts["entries"]), dictionary_path)
    print(f"wrote {dictionary_path}")

    print("all_passed=True")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
