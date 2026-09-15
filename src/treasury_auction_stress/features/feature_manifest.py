"""Loads and validates `configs/phase_5_features.yml`, this project's
single machine-readable feature contract for Phase 5.

Every predictor column actually materialized by
`treasury_auction_stress.features.feature_matrix` must have exactly one
contract entry with `feature_role: predictor` and a literal
`canonical_feature_name` (never only a `name_pattern`) -- validated by
`assert_predictor_columns_match_contract`, so the contract and the code
can never silently drift apart. `artifacts/phase_5_feature_dictionary.md`
is rendered directly from this module's parsed output -- never
hand-maintained as a second, independent source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONTRACT_PATH = Path("configs/phase_5_features.yml")

VALID_FEATURE_ROLES: frozenset[str] = frozenset(
    {
        "identifier",
        "cutoff_metadata",
        "predictor",
        "audit_provenance",
        "target",
        "target_diagnostic",
        "target_construction_only",
        "excluded",
    }
)
VALID_TIERS: frozenset[str] = frozenset({"core", "extended", "audit_only", "excluded"})

REQUIRED_ENTRY_KEYS: tuple[str, ...] = (
    "source_family",
    "source_field_or_derivation",
    "description",
    "economic_rationale",
    "transformation",
    "unit",
    "dtype",
    "applicable_forecast_cutoff",
    "required_lookback",
    "minimum_history",
    "availability_provenance",
    "missingness_meaning",
    "schema_period_limitations",
    "feature_role",
    "default_tier",
    "leakage_risk",
    "leakage_mitigation",
)


@dataclass(frozen=True)
class FeatureEntry:
    """One parsed contract entry. `name` is either a literal
    `canonical_feature_name` or a `name_pattern` string (distinguished
    by `is_pattern`).
    """

    name: str
    is_pattern: bool
    feature_role: str
    default_tier: str
    fields: dict[str, Any]


def _validate_entry(raw: dict[str, Any], *, index: int) -> None:
    has_literal = "canonical_feature_name" in raw
    has_pattern = "name_pattern" in raw
    if has_literal == has_pattern:
        raise ValueError(
            f"phase_5_features.yml entry #{index}: exactly one of "
            "'canonical_feature_name' or 'name_pattern' must be set"
        )
    missing = [k for k in REQUIRED_ENTRY_KEYS if k not in raw]
    if missing:
        name = raw.get("canonical_feature_name") or raw.get("name_pattern")
        raise ValueError(f"phase_5_features.yml entry {name!r}: missing required keys {missing}")
    if raw["feature_role"] not in VALID_FEATURE_ROLES:
        raise ValueError(
            f"phase_5_features.yml entry #{index}: invalid feature_role {raw['feature_role']!r}, "
            f"must be one of {sorted(VALID_FEATURE_ROLES)}"
        )
    if raw["default_tier"] not in VALID_TIERS:
        raise ValueError(
            f"phase_5_features.yml entry #{index}: invalid default_tier {raw['default_tier']!r}, "
            f"must be one of {sorted(VALID_TIERS)}"
        )
    if has_literal and raw["feature_role"] == "predictor" and raw["default_tier"] not in ("core", "extended"):
        raise ValueError(
            f"phase_5_features.yml entry {raw['canonical_feature_name']!r}: a predictor's "
            f"default_tier must be 'core' or 'extended', got {raw['default_tier']!r}"
        )


def load_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Parse and validate the raw YAML contract. Raises `ValueError` on
    any structural problem (missing required key, invalid role/tier,
    both or neither of canonical_feature_name/name_pattern set) -- this
    project never silently tolerates a malformed contract entry.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = raw.get("features", [])
    for i, entry in enumerate(entries):
        _validate_entry(entry, index=i)
    return raw


def parse_entries(contract: dict[str, Any]) -> list[FeatureEntry]:
    """Turn the raw parsed YAML into a list of `FeatureEntry` objects."""
    out = []
    for raw in contract.get("features", []):
        if "canonical_feature_name" in raw:
            name, is_pattern = raw["canonical_feature_name"], False
        else:
            name, is_pattern = raw["name_pattern"], True
        out.append(
            FeatureEntry(
                name=name,
                is_pattern=is_pattern,
                feature_role=raw["feature_role"],
                default_tier=raw["default_tier"],
                fields=raw,
            )
        )
    return out


def literal_predictor_names(
    entries: list[FeatureEntry], *, exclude_source_families: frozenset[str] = frozenset()
) -> dict[str, str]:
    """`{canonical_feature_name: default_tier}` for every literal
    (non-pattern) entry with `feature_role == "predictor"`. Used to
    validate the actual predictor-matrix columns against the contract.
    `exclude_source_families` filters out entries belonging to a source
    family that lives outside the two main predictor matrices (e.g.
    `pre_auction_incremental`, whose columns belong to the separate
    pre-auction update table).
    """
    return {
        e.name: e.default_tier
        for e in entries
        if not e.is_pattern
        and e.feature_role == "predictor"
        and e.fields.get("source_family") not in exclude_source_families
    }


def literal_target_names(entries: list[FeatureEntry]) -> dict[str, str]:
    """`{canonical_feature_name: feature_role}` for every literal entry
    whose role is `target` or `target_diagnostic`.
    """
    return {
        e.name: e.feature_role
        for e in entries
        if not e.is_pattern and e.feature_role in ("target", "target_diagnostic")
    }


def assert_predictor_columns_match_contract(
    columns: list[str], entries: list[FeatureEntry], *, tier: str | None = None
) -> None:
    """Raise `AssertionError` if `columns` (the actual predictor-matrix
    column set, excluding identifier/cutoff-metadata columns) and the
    contract's literal predictor entries (optionally filtered to one
    `tier`) disagree in either direction -- a column in the matrix with
    no contract entry, or a contract predictor entry never actually
    produced, are both treated as a Phase 5 contract-drift bug.
    """
    contract_predictors = literal_predictor_names(entries)
    if tier is not None:
        contract_predictors = {name: t for name, t in contract_predictors.items() if t == tier}
    contract_set = set(contract_predictors)
    column_set = set(columns)
    missing_from_matrix = contract_set - column_set
    missing_from_contract = column_set - contract_set
    if missing_from_matrix or missing_from_contract:
        raise AssertionError(
            "Predictor matrix columns and configs/phase_5_features.yml disagree.\n"
            f"In contract but not in matrix: {sorted(missing_from_matrix)}\n"
            f"In matrix but not in contract: {sorted(missing_from_contract)}"
        )


def render_feature_dictionary_markdown(contract: dict[str, Any], entries: list[FeatureEntry]) -> str:
    """Render `artifacts/phase_5_feature_dictionary.md` directly from the
    parsed contract -- the only source of the dictionary's content.
    """
    meta = contract.get("metadata", {})
    lines = [
        "# Phase 5 Feature Dictionary",
        "",
        (
            "Generated by `treasury_auction_stress.features.feature_manifest` "
            "directly from `configs/phase_5_features.yml` -- this file is never "
            "hand-edited independently of that contract."
        ),
        "",
        f"**Status**: {meta.get('status', 'n/a')}",
        f"**Auction key**: {meta.get('auction_key_definition', 'n/a')}",
        f"**Sample**: {meta.get('sample', 'n/a')}",
        "",
        (
            "A `name_pattern` row (curly-brace notation, e.g. "
            "`{ROUTPUT,RUC,EMPLOY}_level`) describes a *family* of columns "
            "sharing one construction and one risk profile, rather than "
            "forcing needless repetition across near-identical per-series or "
            "per-macro-variable columns. Every predictor and target column "
            "actually materialized in the Phase 5 matrices has its own "
            "literal name entry, never only a pattern."
        ),
        "",
    ]

    role_order = [
        "identifier",
        "cutoff_metadata",
        "predictor",
        "target",
        "target_diagnostic",
        "target_construction_only",
        "audit_provenance",
        "excluded",
    ]
    role_titles = {
        "identifier": "Identifiers",
        "cutoff_metadata": "Cutoff metadata",
        "predictor": "Predictors",
        "target": "Targets",
        "target_diagnostic": "Target diagnostics",
        "target_construction_only": "Target-construction-only",
        "audit_provenance": "Audit / provenance",
        "excluded": "Deliberately excluded",
    }
    tier_order = ["core", "extended", "audit_only", "excluded"]

    for role in role_order:
        role_entries = [e for e in entries if e.feature_role == role]
        if not role_entries:
            continue
        lines.append(f"## {role_titles[role]}")
        lines.append("")
        if role == "predictor":
            for tier in tier_order:
                tier_entries = [e for e in role_entries if e.default_tier == tier]
                if not tier_entries:
                    continue
                lines.append(f"### Tier: {tier} ({len(tier_entries)} predictors)")
                lines.append("")
                lines += _render_entry_table(tier_entries)
                lines.append("")
        else:
            lines += _render_entry_table(role_entries)
            lines.append("")

    return "\n".join(lines) + "\n"


def _render_entry_table(entries: list[FeatureEntry]) -> list[str]:
    lines = [
        "| Name | Source family | Description | Economic rationale | Unit | Cutoff | Tier | Leakage mitigation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(entries, key=lambda x: x.name):
        f = e.fields
        cutoff = f.get("applicable_forecast_cutoff")
        cutoff_str = ", ".join(cutoff) if isinstance(cutoff, list) else str(cutoff)
        lines.append(
            f"| `{e.name}` | {f.get('source_family', '')} | {f.get('description', '')} | "
            f"{f.get('economic_rationale', '')} | {f.get('unit', '')} | {cutoff_str} | "
            f"{e.default_tier} | {f.get('leakage_mitigation', '')} |"
        )
    return lines
