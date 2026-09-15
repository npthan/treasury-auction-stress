"""Phase 5: build the two point-in-time predictor matrices, the
physically separate target table, per-cutoff audit/provenance tables,
and the pre-auction information-update table.

## Design principle: whitelist, never blacklist

Every intermediate join table this module consumes
(`dealer_join.build_dealer_join_table`,
`phase4d_source_joins.build_rates_join_table`/`build_cftc_join_table`/
`build_rtdsm_join_table`, `auction_candidate_features.
build_auction_structure_feature_table`) may itself carry many columns
beyond the ones this module ultimately selects. The final predictor
matrices are assembled by **explicitly selecting exactly the literal
predictor names declared in `configs/phase_5_features.yml`** (via
`feature_manifest.literal_predictor_names`) -- never by taking "every
column except a forbidden list." A blacklist can miss a renamed field;
a whitelist cannot accidentally include one that was never asked for.
`validate_predictor_matrix` additionally asserts the forbidden-field
list (`phase4d_source_joins.FORBIDDEN_RESULT_COLUMNS`) is absent, as a
second, independent line of defense.

## Row alignment is by explicit auction key, never by row position

**Acceptance-review correction**: an earlier version of this module
combined every per-source table via `pandas.concat(..., axis=1)`,
trusting that every upstream builder happened to preserve the same row
order as the input sample. That trust was well-founded (every as-of
join module here restores its own input's row order via an internal
`__as_of_join_row_order__` column) but was never *itself checked* at
the point of assembly -- a latent risk if any upstream builder's
ordering guarantee were ever weakened, silently misaligning an
auction's identity from its own features without any error.

Every per-source table is now given an explicit `auction_key` column
(`make_auction_key`, derived from that table's own `cusip` +
`auction_date` -- the same stable, deterministic key used everywhere
else in Phase 5) the moment it is built
(`build_join_tables`/`_with_auction_key`), immediately validated
against the canonical sample's own key set
(`_validate_keys_against_canonical` -- raises on any duplicate key or
any set mismatch, never silently truncates or reorders), and combined
into the final matrices/audit tables via an explicit
`pandas.merge(..., on="auction_key", how="left", validate="one_to_one")`
(`_merge_selected`) -- never a positional `concat`. `validate="one_to_one"`
is pandas' own built-in guarantee that the merge key is unique on both
sides; combined with the pre-merge set-equality check, a source table
that is reordered, has a swapped-but-still-unique key, or is missing/
duplicating a key is either corrected by the key-based merge (reorder)
or raised loudly (duplicate/missing/extra key) -- never silently
misaligned. See `tests/test_feature_matrix.py`'s dedicated
shuffled-source and swapped-key tests for direct proof.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from treasury_auction_stress.data.cftc_schema import TENOR_TO_CONTRACT_CODES
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY
from treasury_auction_stress.features.auction_candidate_features import (
    build_auction_structure_feature_table,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)
from treasury_auction_stress.features.dealer_join import build_dealer_join_table
from treasury_auction_stress.features.feature_manifest import (
    FeatureEntry,
    literal_target_names,
)
from treasury_auction_stress.features.phase4d_source_joins import (
    FORBIDDEN_RESULT_COLUMNS,
    build_cftc_join_table,
    build_rates_join_table,
    build_rtdsm_join_table,
)
from treasury_auction_stress.features.targets import (
    DIAGNOSTIC_SHARE_NUMERATOR_FIELDS,
    SHARE_NUMERATOR_FIELDS,
    add_all_targets,
)

AUCTION_KEY_COL = "auction_key"
CUTOFF_ANNOUNCEMENT = "announcement"
CUTOFF_PRE_AUCTION = "pre_auction"
CUTOFF_COL_BY_NAME: dict[str, str] = {
    CUTOFF_ANNOUNCEMENT: ANNOUNCEMENT_CUTOFF_COL,
    CUTOFF_PRE_AUCTION: PRE_AUCTION_CUTOFF_COL,
}
INCREMENTAL_SOURCE_FAMILY = "pre_auction_incremental"

# Identifier + cutoff-metadata columns present, in this order, at the
# front of every predictor matrix and every audit table.
ID_METADATA_COLS: tuple[str, ...] = (
    AUCTION_KEY_COL,
    "cusip",
    "auction_date",
    "announcemt_date",
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)

# The auction-structure fields the per-source join-table builders need
# on their input frame (offering_amt for the dealer-inventory ratio;
# tenor/is_reopening/cusip/announcemt_date/auction_date for identity
# and cutoffs). Fetched from the eligible sample once, up front.
_BASE_INPUT_COLS: tuple[str, ...] = (
    "cusip",
    "tenor",
    "auction_date",
    "announcemt_date",
    "is_reopening",
    "offering_amt",
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
)

# Step 12: the small, pre-specified set of core numeric features whose
# announcement-vs-pre-auction VALUE change is materialized in the
# pre-auction update table -- never a mechanical diff of every column.
CORE_CHANGE_FEATURES: tuple[str, ...] = (
    "matched_tenor_par_yield_percent",
    "slope_2s10s_bps",
    "slope_5s30s_bps",
    "matched_dealer_net_contracts",
    "dealer_net_position_total_ex_tips_harmonized",
)


def make_auction_key(df: pd.DataFrame) -> pd.Series:
    """Stable, deterministic auction key: CUSIP + auction date -- never
    row position. Matches `normalize.CANDIDATE_PRIMARY_KEY`.
    """
    return df["cusip"].astype(str) + "_" + pd.to_datetime(df["auction_date"]).dt.date.astype(str)


def _with_auction_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with an explicit `auction_key` column
    computed from its own `cusip`/`auction_date` columns. Every
    per-source table gets this called on it exactly once, immediately
    after it is built, so every later combination step can align by
    key instead of trusting row order.
    """
    out = df.copy()
    out[AUCTION_KEY_COL] = make_auction_key(out)
    return out


def _validate_keys_against_canonical(table: pd.DataFrame, *, canonical_keys: frozenset[str], table_label: str) -> None:
    """Raise `ValueError` if `table[AUCTION_KEY_COL]` has any duplicate,
    or if its key *set* is not exactly `canonical_keys`. Never silently
    drops, truncates, or reorders -- a source table that disagrees with
    the canonical sample's own auction-key set is always a bug to
    surface, not a size mismatch to tolerate.
    """
    keys = table[AUCTION_KEY_COL]
    if keys.duplicated().any():
        dupes = sorted(keys[keys.duplicated(keep=False)].unique())
        raise ValueError(f"{table_label}: duplicate auction keys found: {dupes[:10]}{'...' if len(dupes) > 10 else ''}")
    key_set = frozenset(keys)
    missing = canonical_keys - key_set
    extra = key_set - canonical_keys
    if missing or extra:
        raise ValueError(
            f"{table_label}: auction key set does not match the canonical sample's key set.\n"
            f"Missing from {table_label} ({len(missing)}): {sorted(missing)[:10]}\n"
            f"Present in {table_label} but not the canonical sample ({len(extra)}): {sorted(extra)[:10]}"
        )


def _merge_selected(
    base: pd.DataFrame, source_table: pd.DataFrame, *, source_columns: list[str], rename_to: list[str] | None, table_label: str
) -> pd.DataFrame:
    """Explicitly key-based combination step: select `source_columns`
    (optionally renamed to `rename_to`) plus `AUCTION_KEY_COL` from
    `source_table`, validate the key set against `base`'s own, and
    `merge(..., on=AUCTION_KEY_COL, how="left", validate="one_to_one")`
    onto `base` -- never a positional `concat`. `validate="one_to_one"`
    is pandas' own enforcement that the join key is unique on both
    sides, layered on top of this project's own explicit pre-checks
    (clearer error messages, and a check that catches a duplicate
    *before* pandas would).
    """
    if not source_columns:
        return base
    if AUCTION_KEY_COL not in source_table.columns:
        raise KeyError(f"{table_label}: missing '{AUCTION_KEY_COL}' -- call _with_auction_key on it first")

    selection = source_table[[AUCTION_KEY_COL, *source_columns]].copy()
    if rename_to is not None:
        selection = selection.rename(columns=dict(zip(source_columns, rename_to, strict=True)))

    _validate_keys_against_canonical(selection, canonical_keys=frozenset(base[AUCTION_KEY_COL]), table_label=table_label)

    merged = base.merge(selection, on=AUCTION_KEY_COL, how="left", validate="one_to_one")
    if len(merged) != len(base):
        raise AssertionError(f"{table_label}: merge unexpectedly changed row count from {len(base)} to {len(merged)}")
    return merged


@dataclass(frozen=True)
class SourceTables:
    """Every per-source processed/join table Phase 5 needs, already
    loaded from `data/processed/`. Kept as one small container so the
    CLI and tests can build it once and pass it around.
    """

    sample: pd.DataFrame  # ordinary_modeling_sample, with cutoff dates already added
    dealer_wide: pd.DataFrame
    rates_wide: pd.DataFrame
    cftc_wide: pd.DataFrame
    rtdsm_vintage_indices: dict[str, pd.DataFrame]
    rtdsm_snapshots: dict[str, pd.DataFrame]


def _predictor_names_by_family(
    entries: list[FeatureEntry], *, exclude_source_families: frozenset[str] = frozenset()
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for e in entries:
        if e.feature_role != "predictor" or e.is_pattern:
            continue
        family = e.fields.get("source_family")
        if family in exclude_source_families:
            continue
        groups.setdefault(family, []).append(e.name)
    return groups


def _cutoff_prefixed_columns(names: list[str], cutoff_col: str) -> tuple[list[str], list[str]]:
    """`(source_columns, rename_to)` for `_merge_selected`: every name
    prefixed `{cutoff_col}__`, renamed back to the bare name.
    """
    return [f"{cutoff_col}__{n}" for n in names], list(names)


def build_join_tables(tables: SourceTables) -> dict[str, pd.DataFrame]:
    """Build all four per-source join tables once, from one
    deterministically-sorted copy of the sample. Every returned table
    (including `sample` itself) carries an explicit, validated
    `auction_key` column -- downstream assembly (`build_predictor_matrix`,
    `build_audit_table`) aligns by that key, never by row position.
    """
    sample = _with_auction_key(tables.sample.sort_values(["cusip", "auction_date"]).reset_index(drop=True))
    canonical_keys = frozenset(sample[AUCTION_KEY_COL])
    if len(canonical_keys) != len(sample):
        raise ValueError("build_join_tables: the input sample itself has duplicate auction keys")

    base = sample[list(_BASE_INPUT_COLS)].copy()

    auction_structure = _with_auction_key(build_auction_structure_feature_table(base))
    _validate_keys_against_canonical(auction_structure, canonical_keys=canonical_keys, table_label="auction_structure")

    dealer_join_table = _with_auction_key(build_dealer_join_table(sample, tables.dealer_wide))
    _validate_keys_against_canonical(dealer_join_table, canonical_keys=canonical_keys, table_label="dealer_join_table")

    rates_join_table = _with_auction_key(build_rates_join_table(sample, tables.rates_wide))
    _validate_keys_against_canonical(rates_join_table, canonical_keys=canonical_keys, table_label="rates_join_table")

    cftc_join_table = _with_auction_key(build_cftc_join_table(sample, tables.cftc_wide))
    _validate_keys_against_canonical(cftc_join_table, canonical_keys=canonical_keys, table_label="cftc_join_table")

    rtdsm_join_table = _with_auction_key(build_rtdsm_join_table(sample, tables.rtdsm_vintage_indices, tables.rtdsm_snapshots))
    _validate_keys_against_canonical(rtdsm_join_table, canonical_keys=canonical_keys, table_label="rtdsm_join_table")

    return {
        "sample": sample,
        "auction_structure": auction_structure,
        "dealer": dealer_join_table,
        "rates": rates_join_table,
        "cftc": cftc_join_table,
        "rtdsm": rtdsm_join_table,
    }


def build_predictor_matrix(
    *, cutoff_name: str, joined: dict[str, pd.DataFrame], entries: list[FeatureEntry]
) -> pd.DataFrame:
    """Build one predictor matrix (`cutoff_name` in
    `{"announcement", "pre_auction"}`) from the tables returned by
    `build_join_tables`. Both matrices share the identical column set
    (only the underlying as-of-joined VALUES differ) and deterministic
    (sorted) column order.

    Every per-source contribution is combined by an explicit
    `auction_key`-based merge (`_merge_selected`), never by positional
    concatenation -- see the module docstring's "Row alignment" section.
    """
    if cutoff_name not in CUTOFF_COL_BY_NAME:
        raise ValueError(f"cutoff_name must be one of {list(CUTOFF_COL_BY_NAME)}, got {cutoff_name!r}")
    cutoff_col = CUTOFF_COL_BY_NAME[cutoff_name]
    sample = joined["sample"]
    groups = _predictor_names_by_family(entries, exclude_source_families=frozenset({INCREMENTAL_SOURCE_FAMILY}))

    matrix = pd.DataFrame(
        {
            AUCTION_KEY_COL: sample[AUCTION_KEY_COL].to_numpy(),
            "cusip": sample["cusip"].to_numpy(),
            "auction_date": sample["auction_date"].to_numpy(),
            "announcemt_date": sample["announcemt_date"].to_numpy(),
            ANNOUNCEMENT_CUTOFF_COL: sample[ANNOUNCEMENT_CUTOFF_COL].to_numpy(),
            PRE_AUCTION_CUTOFF_COL: sample[PRE_AUCTION_CUTOFF_COL].to_numpy(),
        }
    )

    auction_structure_names = groups.get("auction_structure", [])
    matrix = _merge_selected(
        matrix, joined["auction_structure"], source_columns=auction_structure_names, rename_to=None, table_label="auction_structure"
    )
    for table_key, family, table_label in (
        ("dealer", "dealer_stats", "dealer_join_table"),
        ("rates", "treasury_rates", "rates_join_table"),
        ("cftc", "cftc_positioning", "cftc_join_table"),
        ("rtdsm", "rtdsm_macro", "rtdsm_join_table"),
    ):
        names = groups.get(family, [])
        if not names:
            continue
        source_columns, rename_to = _cutoff_prefixed_columns(names, cutoff_col)
        missing = [c for c in source_columns if c not in joined[table_key].columns]
        if missing:
            raise KeyError(
                f"{table_label}: configs/phase_5_features.yml names predictors for cutoff "
                f"'{cutoff_col}' that this table does not produce: {missing}"
            )
        matrix = _merge_selected(matrix, joined[table_key], source_columns=source_columns, rename_to=rename_to, table_label=table_label)

    predictor_cols = sorted(c for c in matrix.columns if c not in ID_METADATA_COLS)
    matrix = matrix[list(ID_METADATA_COLS) + predictor_cols].sort_values(AUCTION_KEY_COL).reset_index(drop=True)
    return matrix


def validate_predictor_matrix(matrix: pd.DataFrame, entries: list[FeatureEntry], *, expected_rows: int) -> None:
    """Fail loudly (raise `AssertionError`) on any of: wrong row count,
    duplicate auction keys, a forbidden result column present, or a
    mismatch between the matrix's own predictor columns and the
    contract's literal main-matrix predictor names.
    """
    if len(matrix) != expected_rows:
        raise AssertionError(f"predictor matrix has {len(matrix)} rows, expected {expected_rows}")
    if matrix[AUCTION_KEY_COL].duplicated().any():
        dupes = matrix.loc[matrix[AUCTION_KEY_COL].duplicated(keep=False), AUCTION_KEY_COL].unique()
        raise AssertionError(f"predictor matrix has duplicate auction keys: {sorted(dupes)[:10]}")
    if matrix[AUCTION_KEY_COL].isna().any() or matrix["cusip"].isna().any() or matrix["auction_date"].isna().any():
        raise AssertionError("predictor matrix has a null identifier value")

    leaked = sorted(set(matrix.columns) & set(FORBIDDEN_RESULT_COLUMNS))
    if leaked:
        raise AssertionError(f"predictor matrix contains forbidden result columns: {leaked}")

    predictor_cols = [c for c in matrix.columns if c not in ID_METADATA_COLS]
    from treasury_auction_stress.features.feature_manifest import (
        assert_predictor_columns_match_contract,
    )

    assert_predictor_columns_match_contract(
        predictor_cols,
        [e for e in entries if e.fields.get("source_family") != INCREMENTAL_SOURCE_FAMILY],
    )


def build_target_table(sample: pd.DataFrame) -> pd.DataFrame:
    """The physically separate target table: auction_key plus the four
    accepted Phase 2 raw outcomes and five clearly-marked diagnostics.
    Never merged into a predictor matrix by this module -- callers must
    merge explicitly, keyed by `auction_key`.
    """
    with_targets = add_all_targets(sample)
    out = pd.DataFrame({AUCTION_KEY_COL: make_auction_key(with_targets)})
    for col in (*SHARE_NUMERATOR_FIELDS, *DIAGNOSTIC_SHARE_NUMERATOR_FIELDS):
        out[col] = with_targets[col].to_numpy()
    out["bid_to_cover_ratio"] = with_targets["bid_to_cover_ratio"].to_numpy()
    out["public_accepted_amount"] = with_targets["public_accepted_amount"].to_numpy()
    out["bid_to_cover_calculated"] = with_targets["bid_to_cover_calculated"].to_numpy()
    out["bid_to_cover_reconciliation_diff"] = with_targets["bid_to_cover_reconciliation_diff"].to_numpy()
    return out.sort_values(AUCTION_KEY_COL).reset_index(drop=True)


def validate_target_table(target_table: pd.DataFrame, entries: list[FeatureEntry], *, expected_rows: int) -> None:
    if len(target_table) != expected_rows:
        raise AssertionError(f"target table has {len(target_table)} rows, expected {expected_rows}")
    if target_table[AUCTION_KEY_COL].duplicated().any():
        raise AssertionError("target table has duplicate auction keys")
    contract_targets = set(literal_target_names(entries))
    actual = set(target_table.columns) - {AUCTION_KEY_COL}
    if actual != contract_targets:
        raise AssertionError(
            "target table columns and configs/phase_5_features.yml disagree.\n"
            f"In contract but not in table: {sorted(contract_targets - actual)}\n"
            f"In table but not in contract: {sorted(actual - contract_targets)}"
        )


def build_audit_table(*, cutoff_name: str, joined: dict[str, pd.DataFrame], entries: list[FeatureEntry]) -> pd.DataFrame:
    """Every source-specific provenance column for one cutoff, keyed
    1:1 to that cutoff's predictor matrix: literally every `{cutoff}__`
    column the dealer/rates/cftc/rtdsm join tables produce that is
    *not* one of the predictor names already selected into the matrix
    -- i.e. every join-matched flag, observation age, missing reason,
    availability-precision label, vintage identity, and source-gap
    flag those four join modules compute. Raises on an unexpected
    cross-source column-name collision (fail loudly, never silently
    overwrite). Combined via an explicit `auction_key`-based merge,
    never positional concatenation -- see the module docstring.
    """
    cutoff_col = CUTOFF_COL_BY_NAME[cutoff_name]
    sample = joined["sample"]
    groups = _predictor_names_by_family(entries, exclude_source_families=frozenset({INCREMENTAL_SOURCE_FAMILY}))

    audit = pd.DataFrame(
        {
            AUCTION_KEY_COL: sample[AUCTION_KEY_COL].to_numpy(),
            "cusip": sample["cusip"].to_numpy(),
            "auction_date": sample["auction_date"].to_numpy(),
        }
    )
    seen_cols: set[str] = set(audit.columns)

    for table_key, family in (("dealer", "dealer_stats"), ("rates", "treasury_rates"), ("cftc", "cftc_positioning"), ("rtdsm", "rtdsm_macro")):
        table = joined[table_key]
        predictor_names = set(groups.get(family, []))
        prefix = f"{cutoff_col}__"
        audit_cols = [c for c in table.columns if c.startswith(prefix) and c[len(prefix):] not in predictor_names]
        # Namespaced by source family: several sources independently
        # produce identically-named provenance columns (e.g. both
        # dealer stats and Treasury rates carry their own
        # `publication_date`/`publication_safe_available_date`) --
        # prefixing here guarantees no cross-source collision, rather
        # than merely detecting one after the fact.
        rename_to = [f"{family}__{c[len(prefix):]}" for c in audit_cols]
        collisions = seen_cols & set(rename_to)
        if collisions:
            raise AssertionError(f"build_audit_table: unexpected cross-source column collision: {sorted(collisions)}")
        seen_cols |= set(rename_to)
        audit = _merge_selected(audit, table, source_columns=audit_cols, rename_to=rename_to, table_label=f"{table_key}_join_table (audit)")

    ordered = ["auction_key", "cusip", "auction_date"] + sorted(c for c in audit.columns if c not in ("auction_key", "cusip", "auction_date"))
    return audit[ordered].sort_values(AUCTION_KEY_COL).reset_index(drop=True)


def validate_audit_table(audit_table: pd.DataFrame, matrix: pd.DataFrame, *, cutoff_name: str) -> None:
    """**Key-integrity checks only** -- the audit table must be keyed
    1:1 to its predictor matrix (identical key sets, no duplicates).

    This function deliberately does NOT check availability timing
    (acceptance-review correction: an earlier docstring implied it did,
    which it never actually enforced). Timing is checked separately,
    and mandatorily, by `assert_audit_table_availability_safe` below --
    call both, in that order, before publishing any artifact.
    """
    if set(audit_table[AUCTION_KEY_COL]) != set(matrix[AUCTION_KEY_COL]):
        raise AssertionError(f"{cutoff_name} audit table and predictor matrix key sets disagree")
    if audit_table[AUCTION_KEY_COL].duplicated().any():
        raise AssertionError(f"{cutoff_name} audit table has duplicate auction keys")


def assert_audit_table_availability_safe(audit_table: pd.DataFrame, matrix: pd.DataFrame, *, cutoff_col: str) -> dict[str, dict]:
    """**Mandatory, fail-fast timing enforcement** (acceptance-review
    addition): checks the actual artifact about to be published (not
    merely an intermediate join table) for all four source families,
    and raises `AssertionError` on the first family with even one row
    whose selected safe-availability date/vintage is after `cutoff_col`.
    Returns per-family `{n_matched, n_violations}` for reporting when
    every family passes.
    """
    merged = matrix[[AUCTION_KEY_COL, cutoff_col]].merge(audit_table, on=AUCTION_KEY_COL, how="inner", validate="one_to_one")
    if len(merged) != len(matrix):
        raise AssertionError("assert_audit_table_availability_safe: audit table and matrix key sets disagree")
    cutoff = merged[cutoff_col]

    results: dict[str, dict] = {}
    for family, safe_col in (
        ("dealer_stats", "publication_safe_available_date"),
        ("treasury_rates", "publication_safe_available_date"),
        ("cftc_positioning", "publication_safe_available_date"),
    ):
        col = f"{family}__{safe_col}"
        if col not in merged.columns:
            raise KeyError(f"assert_audit_table_availability_safe: expected audit column {col!r} not found")
        safe_dates = merged[col]
        matched = safe_dates.notna()
        violations = matched & (safe_dates > cutoff)
        results[family] = {"n_matched": int(matched.sum()), "n_violations": int(violations.sum())}
        if violations.any():
            bad_keys = sorted(merged.loc[violations, AUCTION_KEY_COL])[:10]
            raise AssertionError(
                f"assert_audit_table_availability_safe: {family} has {int(violations.sum())} row(s) "
                f"whose safe-availability date is AFTER the cutoff (e.g. {bad_keys})"
            )

    for variable in VARIABLE_REGISTRY:
        m = variable.mnemonic
        lag_col = f"rtdsm_macro__{m}_vintage_lag_calendar_days"
        matched_col = f"rtdsm_macro__{m}_join_matched"
        if lag_col not in merged.columns or matched_col not in merged.columns:
            raise KeyError(f"assert_audit_table_availability_safe: expected audit columns {lag_col!r}/{matched_col!r} not found")
        lag = merged[lag_col]
        matched = merged[matched_col].fillna(False).astype(bool)
        # lag = cutoff - safe_date; a negative lag means the safe date is AFTER the cutoff.
        violations = matched & (lag < 0)
        results[f"rtdsm_{m}"] = {"n_matched": int(matched.sum()), "n_violations": int(violations.sum())}
        if violations.any():
            bad_keys = sorted(merged.loc[violations, AUCTION_KEY_COL])[:10]
            raise AssertionError(
                f"assert_audit_table_availability_safe: rtdsm {m} has {int(violations.sum())} row(s) "
                f"whose selected vintage is AFTER the cutoff (e.g. {bad_keys})"
            )

    return results


def _release_identity_changed(join_table: pd.DataFrame, id_col: str) -> pd.Series:
    """`True` iff the two cutoffs' as-of joins selected a different
    release identity for `id_col` (e.g. `observation_date`,
    `rate_date`, `report_date`, or an RTDSM `{m}_vintage_label`);
    `False` if neither cutoff matched, or both matched the identical
    release; `True` if the pre-auction cutoff matched something the
    announcement cutoff did not. Equality/inequality is sufficient
    (never an ordering comparison) because cross-cutoff monotonicity
    (verified in tests/test_feature_matrix.py) guarantees the
    pre-auction cutoff can never select an *earlier* release than the
    announcement cutoff.
    """
    ann = join_table[f"{ANNOUNCEMENT_CUTOFF_COL}__{id_col}"]
    pre = join_table[f"{PRE_AUCTION_CUTOFF_COL}__{id_col}"]
    ann_present, pre_present = ann.notna(), pre.notna()
    result = pd.array([pd.NA] * len(join_table), dtype="boolean")
    result = pd.Series(result, index=join_table[AUCTION_KEY_COL])
    result.loc[~pre_present.to_numpy()] = False
    result.loc[(pre_present & ~ann_present).to_numpy()] = True
    both = (pre_present & ann_present).to_numpy()
    result.loc[both] = (ann.loc[both].astype(str) != pre.loc[both].astype(str)).to_numpy()
    return result.astype("boolean")


def build_pre_auction_update_table(
    *,
    joined: dict[str, pd.DataFrame],
    announcement_matrix: pd.DataFrame,
    pre_auction_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Step 12: a small, pre-specified table of what changed between
    the announcement and pre-auction cutoffs -- never a mechanical
    subtraction of every numeric column. See
    `configs/phase_5_features.yml`'s `pre_auction_incremental` entries.
    """
    sample = joined["sample"]
    out = pd.DataFrame({AUCTION_KEY_COL: sample[AUCTION_KEY_COL].to_numpy()})
    out["cutoff_gap_calendar_days"] = (
        sample[PRE_AUCTION_CUTOFF_COL] - sample[ANNOUNCEMENT_CUTOFF_COL]
    ).dt.days.to_numpy()
    out_keys = out[AUCTION_KEY_COL]

    # Every _release_identity_changed result is indexed by its own
    # source table's auction_key -- explicitly reindexed (never
    # positionally assumed) to out's own key order.
    out["dealer_stats_new_release_between_cutoffs"] = (
        _release_identity_changed(joined["dealer"], "observation_date").reindex(out_keys).to_numpy()
    )
    out["treasury_rates_new_release_between_cutoffs"] = (
        _release_identity_changed(joined["rates"], "rate_date").reindex(out_keys).to_numpy()
    )
    out["cftc_positioning_new_release_between_cutoffs"] = (
        _release_identity_changed(joined["cftc"], "report_date").reindex(out_keys).to_numpy()
    )
    for variable in VARIABLE_REGISTRY:
        m = variable.mnemonic
        out[f"{m}_new_vintage_between_cutoffs"] = (
            _release_identity_changed(joined["rtdsm"], f"{m}_vintage_label").reindex(out_keys).to_numpy()
        )

    ann_indexed = announcement_matrix.set_index(AUCTION_KEY_COL)
    pre_indexed = pre_auction_matrix.set_index(AUCTION_KEY_COL)
    for feature in CORE_CHANGE_FEATURES:
        diff = pre_indexed[feature].reindex(out_keys).to_numpy() - ann_indexed[feature].reindex(out_keys).to_numpy()
        out[f"{feature}_change_between_cutoffs"] = diff

    return out.sort_values(AUCTION_KEY_COL).reset_index(drop=True)


def validate_pre_auction_update_table(update_table: pd.DataFrame, matrix: pd.DataFrame) -> None:
    if set(update_table[AUCTION_KEY_COL]) != set(matrix[AUCTION_KEY_COL]):
        raise AssertionError("pre-auction update table and predictor matrix key sets disagree")
    if update_table[AUCTION_KEY_COL].duplicated().any():
        raise AssertionError("pre-auction update table has duplicate auction keys")
    if (update_table["cutoff_gap_calendar_days"] < 0).any():
        raise AssertionError("cutoff_gap_calendar_days is negative for at least one auction")


def assert_no_direct_contract_tenor_is_structural(sample: pd.DataFrame) -> None:
    """Sanity check used by the CLI/tests: the 3-Year/7-Year tenors
    (which never have any CFTC futures contract at all) are a real,
    verified absence, not an accidental empty mapping.
    """
    covered = set(TENOR_TO_CONTRACT_CODES)
    all_tenors = set(sample["tenor"].unique())
    uncovered = all_tenors - covered
    if uncovered - {"3-Year", "7-Year"}:
        raise AssertionError(f"unexpected tenor(s) with no CFTC contract mapping: {uncovered}")


def compute_content_digest(df: pd.DataFrame) -> str:
    """Acceptance-review addition (Issue 5): a stable SHA-256 digest of
    a dataframe's full content -- column names (in order), each
    column's own dtype, row order, and every value -- computed over a
    single, fully documented canonical string:

        "{col1}:{dtype1}\\x1f{col2}:{dtype2}\\x1f...\\x1e{csv body}"

    where the CSV body is `DataFrame.to_csv(index=False, header=False)`
    with an explicit, unambiguous missing-value sentinel
    (`"\\x00NA\\x00"`, a control-character sequence that cannot appear
    in this project's own string columns) so a genuinely missing value
    is never confused with the literal text "NA" a real field might
    contain. This replaces the previous `pandas.util.hash_pandas_object(...).sum()`
    fingerprint, which is a SUM of per-row hashes -- order-insensitive
    by construction (two dataframes with the same rows in a different
    order, or even two different rows whose hashes happen to sum to
    the same total, produce an identical "fingerprint") and therefore
    not a meaningful content digest. This digest changes if the column
    order, any dtype, any single value, or the row order changes --
    verified directly in `tests/test_feature_matrix.py`.
    """
    header = "\x1f".join(f"{col}:{df[col].dtype}" for col in df.columns)
    body = df.to_csv(index=False, header=False, na_rep="\x00NA\x00")
    canonical = header + "\x1e" + body
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogatepass")).hexdigest()
