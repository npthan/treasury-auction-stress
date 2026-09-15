"""Phase 5: the adversarial leakage-audit suite.

Every function here is called both by `tests/test_feature_matrix.py`
(so a genuine regression fails the test suite) and by
`treasury_auction_stress.features.feature_matrix_cli` when it
regenerates `artifacts/phase_5_leakage_audit.md` -- the report always
reflects checks that were actually run this session, never a hand-typed
claim, per `docs/project_rules.md`'s "no fabricated results" rule.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd

from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.feature_matrix import (
    AUCTION_KEY_COL,
    CUTOFF_COL_BY_NAME,
    SourceTables,
    build_join_tables,
    build_predictor_matrix,
)
from treasury_auction_stress.features.phase4d_source_joins import (
    FORBIDDEN_RESULT_COLUMNS,
)

# The 5 real, independent inputs a Phase 5 matrix is built from: the
# auction sample itself, plus each of the 4 external source tables.
# Acceptance-review addition (Issue 2): every adversarial check in this
# module that claims to "perturb the sources" must actually perturb
# each of these 5 independently, not merely the auction sample.
PERTURBABLE_SOURCES: tuple[str, ...] = ("sample", "dealer_wide", "rates_wide", "cftc_wide", "rtdsm")

# Case-insensitive substrings that would plausibly appear in a renamed
# auction-RESULT field. Deliberately broad (a defense against a renamed
# obvious result field slipping through a static list), so this is
# checked against the predictor-matrix column set only -- not against
# the source-specific audit tables, which legitimately carry timing
# words like "age" and "date" that would otherwise false-positive here.
FORBIDDEN_NAME_PATTERNS: tuple[str, ...] = (
    r"yield", r"tendered", r"accepted", r"\bprice\b", r"discnt", r"investment_rate",
    r"bid_to_cover", r"allocation_pct", r"accrued_int", r"soma_", r"fima_noncomp",
    r"comp_accepted", r"comp_tendered", r"noncomp_accepted", r"primary_dealer_accepted",
    r"direct_bidder_accepted", r"indirect_bidder_accepted", r"total_accepted", r"total_tendered",
    r"results_available", r"dealer_absorption_surprise", r"stress_threshold", r"stress_label",
    r"stress_event",
)
# Legitimate predictor/audit columns that happen to contain a
# substring above for an unrelated reason -- reviewed individually and
# confirmed not to be a result field.
FORBIDDEN_PATTERN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "matched_tenor_par_yield_percent", "adjacent_lower_par_yield_percent", "adjacent_higher_par_yield_percent",
        "matched_tenor_chg_1d_bps", "matched_tenor_chg_5d_bps", "matched_tenor_chg_20d_bps",
        "matched_tenor_vol_5d_bps", "matched_tenor_vol_20d_bps",
        "matched_tenor_par_yield_percent_change_between_cutoffs",
    }
)


def forbidden_field_scan(columns: list[str]) -> dict[str, list[str]]:
    """Two independent checks against `columns` (a predictor matrix's
    own column list): an exact match against
    `phase4d_source_joins.FORBIDDEN_RESULT_COLUMNS`, and a case-
    insensitive substring/pattern scan (minus the reviewed allowlist)
    that would catch a renamed-but-still-a-result field even if it
    were absent from the static list.
    """
    exact = sorted(set(columns) & set(FORBIDDEN_RESULT_COLUMNS))
    pattern_hits = []
    for col in columns:
        if col in FORBIDDEN_PATTERN_ALLOWLIST:
            continue
        for pattern in FORBIDDEN_NAME_PATTERNS:
            if re.search(pattern, col, flags=re.IGNORECASE):
                pattern_hits.append(col)
                break
    return {"exact_matches": exact, "pattern_matches": sorted(set(pattern_hits))}


def max_safe_availability_lag(joined_table: pd.DataFrame, *, cutoff_col: str, safe_date_col: str) -> dict:
    """For one source's join table and one cutoff: the maximum
    (cutoff - selected safe-availability date) lag among matched rows,
    and proof that no matched row's selected safe-availability date
    exceeds the cutoff. `safe_date_col` is the source's own bare
    `publication_safe_available_date`/`report_date`-style column name
    (unprefixed); this function reads `{cutoff_col}__{safe_date_col}`.
    """
    col = f"{cutoff_col}__{safe_date_col}"
    cutoffs = joined_table[cutoff_col]
    safe_dates = joined_table[col]
    matched = safe_dates.notna()
    violations = matched & (safe_dates > cutoffs)
    lag_days = (cutoffs - safe_dates).dt.days
    return {
        "n_matched": int(matched.sum()),
        "n_violations": int(violations.sum()),
        "max_lag_days": float(lag_days.loc[matched].max()) if matched.any() else None,
        "violating_keys": sorted(joined_table.loc[violations, AUCTION_KEY_COL]) if AUCTION_KEY_COL in joined_table.columns else [],
    }


def assert_safe_availability_never_exceeds_cutoff(joined: dict[str, pd.DataFrame]) -> dict:
    """For all four source families and both cutoffs: prove the
    selected `publication_safe_available_date` (or `report_date`'s own
    safe date, or the matched RTDSM vintage's safe date) never exceeds
    the applicable cutoff. Returns a results dict suitable for
    rendering directly into the leakage report.
    """
    results: dict[str, dict] = {}
    for cutoff_name, cutoff_col in CUTOFF_COL_BY_NAME.items():
        for table_key, safe_col in (
            ("dealer", "publication_safe_available_date"),
            ("rates", "publication_safe_available_date"),
            ("cftc", "publication_safe_available_date"),
        ):
            table = joined[table_key]
            col = f"{cutoff_col}__{safe_col}"
            safe_dates = table[col]
            cutoffs = table[cutoff_col]
            matched = safe_dates.notna()
            violations = matched & (safe_dates > cutoffs)
            results[f"{table_key}__{cutoff_name}"] = {
                "n_matched": int(matched.sum()),
                "n_violations": int(violations.sum()),
                "max_lag_days": float((cutoffs - safe_dates).loc[matched].dt.days.max()) if matched.any() else None,
            }

        rtdsm_table = joined["rtdsm"]
        from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY

        for variable in VARIABLE_REGISTRY:
            m = variable.mnemonic
            col = f"{cutoff_col}__{m}_vintage_lag_calendar_days"
            if col not in rtdsm_table.columns:
                continue
            lag = rtdsm_table[col]
            matched_col = f"{cutoff_col}__{m}_join_matched"
            matched = rtdsm_table[matched_col].fillna(False).astype(bool)
            violations = matched & (lag < 0)
            results[f"rtdsm_{m}__{cutoff_name}"] = {
                "n_matched": int(matched.sum()),
                "n_violations": int(violations.sum()),
                "max_lag_days": float(lag.loc[matched].max()) if matched.any() else None,
            }
    return results


def assert_cross_cutoff_monotonicity(joined: dict[str, pd.DataFrame]) -> dict:
    """The pre-auction cutoff is never earlier than the announcement
    cutoff, so the pre-auction view's selected release identity for
    every source must never be *earlier* than the announcement view's
    -- proven directly (never assumed) for all four source families.
    """
    results: dict[str, dict] = {}

    def _check(table: pd.DataFrame, id_col: str, *, is_date: bool) -> dict:
        ann = table[f"{ANNOUNCEMENT_CUTOFF_COL}__{id_col}"]
        pre = table[f"{PRE_AUCTION_CUTOFF_COL}__{id_col}"]
        both = ann.notna() & pre.notna()
        if is_date:
            violations = both & (pre < ann)
        else:
            # Vintage labels aren't chronologically sortable as plain
            # strings in general; monotonicity for these is checked
            # via the lag/safe-date columns elsewhere. This branch is
            # unused for RTDSM but kept for completeness/documentation.
            violations = pd.Series(False, index=table.index)
        return {"n_compared": int(both.sum()), "n_violations": int(violations.sum())}

    results["dealer_observation_date"] = _check(joined["dealer"], "observation_date", is_date=True)
    results["rates_rate_date"] = _check(joined["rates"], "rate_date", is_date=True)
    results["cftc_report_date"] = _check(joined["cftc"], "report_date", is_date=True)

    from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY

    rtdsm_table = joined["rtdsm"]
    for variable in VARIABLE_REGISTRY:
        m = variable.mnemonic
        ann_lag = rtdsm_table.get(f"{ANNOUNCEMENT_CUTOFF_COL}__{m}_vintage_lag_calendar_days")
        pre_lag = rtdsm_table.get(f"{PRE_AUCTION_CUTOFF_COL}__{m}_vintage_lag_calendar_days")
        if ann_lag is None or pre_lag is None:
            continue
        ann_pub = rtdsm_table[ANNOUNCEMENT_CUTOFF_COL] - pd.to_timedelta(ann_lag, unit="D")
        pre_pub = rtdsm_table[PRE_AUCTION_CUTOFF_COL] - pd.to_timedelta(pre_lag, unit="D")
        both = ann_lag.notna() & pre_lag.notna()
        violations = both & (pre_pub < ann_pub)
        results[f"rtdsm_{m}_vintage"] = {"n_compared": int(both.sum()), "n_violations": int(violations.sum())}

    return results


def outcome_poisoning_test(
    build_matrix_fn, sample: pd.DataFrame, *, outcome_columns: tuple[str, ...]
) -> dict:
    """Replace every named outcome/result column in `sample` with
    extreme or randomized values and prove `build_matrix_fn(sample)`
    (a no-argument-except-sample callable that rebuilds a full
    predictor matrix from a sample dataframe) produces a bit-identical
    result. `outcome_columns` should be every column in
    `phase4d_source_joins.FORBIDDEN_RESULT_COLUMNS` actually present in
    `sample`.
    """
    baseline = build_matrix_fn(sample)

    poisoned = sample.copy()
    rng = np.random.default_rng(0)
    for col in outcome_columns:
        if col not in poisoned.columns:
            continue
        if pd.api.types.is_numeric_dtype(poisoned[col]) or pd.api.types.is_float_dtype(poisoned[col]):
            poisoned[col] = rng.uniform(-1e18, 1e18, size=len(poisoned))
        else:
            poisoned[col] = "POISONED"
    poisoned_result = build_matrix_fn(poisoned)

    try:
        pd.testing.assert_frame_equal(baseline, poisoned_result)
        return {"identical": True, "columns_poisoned": [c for c in outcome_columns if c in sample.columns]}
    except AssertionError as exc:
        return {"identical": False, "error": str(exc), "columns_poisoned": [c for c in outcome_columns if c in sample.columns]}


def input_order_invariance_test(build_matrix_fn, sample: pd.DataFrame) -> dict:
    """Randomly permute `sample`'s (the auction table's own) row order
    and prove the canonicalized (sorted-by-auction_key) output is
    bit-identical.

    **Scope note** (acceptance-review addition): this checks only the
    auction-sample input. It does NOT perturb the dealer/rates/cftc/
    rtdsm source tables -- for that, use `per_source_input_order_invariance`
    below, which independently shuffles all 5 real inputs.
    """
    forward = build_matrix_fn(sample)
    rng = np.random.default_rng(1)
    shuffled = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    backward = build_matrix_fn(shuffled)
    try:
        pd.testing.assert_frame_equal(forward, backward)
        return {"identical": True}
    except AssertionError as exc:
        return {"identical": False, "error": str(exc)}


def future_row_invariance_test(build_matrix_fn, sample: pd.DataFrame, future_row: pd.DataFrame) -> dict:
    """Append a synthetic future-dated auction row, rebuild, and prove
    every pre-existing historical row's predictor values are
    unchanged.

    **Scope note** (acceptance-review addition): this checks only the
    auction-sample input. It does NOT append a future observation to
    the dealer/rates/cftc/rtdsm source tables -- for that, use
    `per_source_future_row_invariance` below.
    """
    before = build_matrix_fn(sample)
    extended_sample = pd.concat([sample, future_row], ignore_index=True)
    after = build_matrix_fn(extended_sample)

    after_indexed = after.set_index(AUCTION_KEY_COL)
    before_indexed = before.set_index(AUCTION_KEY_COL)
    common_keys = before_indexed.index
    try:
        pd.testing.assert_frame_equal(before_indexed, after_indexed.loc[common_keys])
        return {"identical_for_historical_rows": True, "n_historical_rows": len(before)}
    except AssertionError as exc:
        return {"identical_for_historical_rows": False, "error": str(exc)}


# =============================================================================
# Acceptance-review additions (Issue 2): per-source perturbation, covering all
# 5 real inputs independently (the auction sample AND each of the 4 external
# source tables), for BOTH prediction cutoffs.
# =============================================================================


def build_matrix_for_tables(tables: SourceTables, *, cutoff_name: str, entries: list) -> pd.DataFrame:
    """Convenience wrapper: `build_join_tables` + `build_predictor_matrix`
    for one `SourceTables` snapshot and one cutoff. The standard
    "build_matrix_fn" used by every per-source perturbation check below.
    """
    joined = build_join_tables(tables)
    return build_predictor_matrix(cutoff_name=cutoff_name, joined=joined, entries=entries)


def _source_table_row_count(tables: SourceTables, source_name: str) -> int:
    if source_name == "sample":
        return len(tables.sample)
    if source_name == "dealer_wide":
        return len(tables.dealer_wide)
    if source_name == "rates_wide":
        return len(tables.rates_wide)
    if source_name == "cftc_wide":
        return len(tables.cftc_wide)
    if source_name == "rtdsm":
        return sum(len(df) for df in tables.rtdsm_vintage_indices.values())
    raise ValueError(f"unknown source_name {source_name!r}")


def _shuffle_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def _with_source_shuffled(tables: SourceTables, *, source_name: str, rng: np.random.Generator) -> SourceTables:
    """Return a copy of `tables` with ONLY `source_name`'s own row
    order permuted -- every other input untouched.
    """
    if source_name == "sample":
        return replace(tables, sample=_shuffle_rows(tables.sample, rng))
    if source_name == "dealer_wide":
        return replace(tables, dealer_wide=_shuffle_rows(tables.dealer_wide, rng))
    if source_name == "rates_wide":
        return replace(tables, rates_wide=_shuffle_rows(tables.rates_wide, rng))
    if source_name == "cftc_wide":
        return replace(tables, cftc_wide=_shuffle_rows(tables.cftc_wide, rng))
    if source_name == "rtdsm":
        return replace(
            tables,
            rtdsm_vintage_indices={m: _shuffle_rows(df, rng) for m, df in tables.rtdsm_vintage_indices.items()},
            rtdsm_snapshots={m: _shuffle_rows(df, rng) for m, df in tables.rtdsm_snapshots.items()},
        )
    raise ValueError(f"unknown source_name {source_name!r}")


def per_source_input_order_invariance(
    tables: SourceTables,
    entries: list,
    *,
    cutoff_names: tuple[str, ...] = ("announcement", "pre_auction"),
    seed: int = 0,
) -> dict:
    """For each of the 5 real inputs (`PERTURBABLE_SOURCES`) and each
    cutoff: shuffle ONLY that one input's row order (every other input
    untouched), rebuild, and prove the resulting predictor matrix is
    bit-identical to the unperturbed baseline. Reports the exact row
    count of the input actually perturbed, so the leakage report can
    state precisely what was tested -- never a claim that a source was
    perturbed when it was not.
    """
    rng = np.random.default_rng(seed)
    baselines = {c: build_matrix_for_tables(tables, cutoff_name=c, entries=entries) for c in cutoff_names}
    results: dict[str, dict] = {}
    for source_name in PERTURBABLE_SOURCES:
        n_rows = _source_table_row_count(tables, source_name)
        perturbed_tables = _with_source_shuffled(tables, source_name=source_name, rng=rng)
        per_cutoff: dict[str, dict] = {}
        for cutoff_name in cutoff_names:
            result = build_matrix_for_tables(perturbed_tables, cutoff_name=cutoff_name, entries=entries)
            try:
                pd.testing.assert_frame_equal(baselines[cutoff_name], result)
                per_cutoff[cutoff_name] = {"identical": True, "n_rows_perturbed": n_rows}
            except AssertionError as exc:
                per_cutoff[cutoff_name] = {"identical": False, "n_rows_perturbed": n_rows, "error": str(exc)}
        results[source_name] = per_cutoff
    return results


_FAR_FUTURE = pd.Timestamp("2099-06-01")


def _with_future_row_appended(tables: SourceTables, *, source_name: str) -> SourceTables:
    """Return a copy of `tables` with one synthetic, far-future-dated
    row appended to ONLY `source_name` -- every other input untouched.
    """
    if source_name == "sample":
        future_row = tables.sample.iloc[[0]].copy()
        future_row["cusip"] = "ZZFUTURE_PERTURB"
        future_row["auction_date"] = _FAR_FUTURE + pd.Timedelta(days=5)
        future_row["announcemt_date"] = _FAR_FUTURE
        future_row = future_row.drop(columns=[ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL], errors="ignore")
        future_row = add_cutoff_dates(future_row)
        return replace(tables, sample=pd.concat([tables.sample, future_row], ignore_index=True))
    if source_name == "dealer_wide":
        future_row = tables.dealer_wide.iloc[[-1]].copy()
        future_row["observation_date"] = _FAR_FUTURE
        future_row["publication_date"] = _FAR_FUTURE + pd.Timedelta(days=1)
        future_row["publication_safe_available_date"] = _FAR_FUTURE + pd.Timedelta(days=2)
        return replace(tables, dealer_wide=pd.concat([tables.dealer_wide, future_row], ignore_index=True))
    if source_name == "rates_wide":
        future_row = tables.rates_wide.iloc[[-1]].copy()
        future_row["rate_date"] = _FAR_FUTURE
        future_row["publication_date"] = _FAR_FUTURE
        future_row["publication_safe_available_date"] = _FAR_FUTURE + pd.Timedelta(days=1)
        return replace(tables, rates_wide=pd.concat([tables.rates_wide, future_row], ignore_index=True))
    if source_name == "cftc_wide":
        future_row = tables.cftc_wide.iloc[[-1]].copy()
        future_row["report_date"] = _FAR_FUTURE
        future_row["publication_safe_available_date"] = _FAR_FUTURE + pd.Timedelta(days=4)
        return replace(tables, cftc_wide=pd.concat([tables.cftc_wide, future_row], ignore_index=True))
    if source_name == "rtdsm":
        new_vintage_indices, new_snapshots = {}, {}
        for m, df in tables.rtdsm_vintage_indices.items():
            future_row = df.iloc[[-1]].copy()
            future_row["vintage_label"] = "99Q9"
            future_row["nominal_publication_date"] = _FAR_FUTURE
            future_row["publication_safe_available_date"] = _FAR_FUTURE + pd.Timedelta(days=1)
            new_vintage_indices[m] = pd.concat([df, future_row], ignore_index=True)
        for m, df in tables.rtdsm_snapshots.items():
            future_row = df.iloc[[-1]].copy()
            future_row["vintage_label"] = "99Q9"
            new_snapshots[m] = pd.concat([df, future_row], ignore_index=True)
        return replace(tables, rtdsm_vintage_indices=new_vintage_indices, rtdsm_snapshots=new_snapshots)
    raise ValueError(f"unknown source_name {source_name!r}")


def per_source_future_row_invariance(
    tables: SourceTables, entries: list, *, cutoff_names: tuple[str, ...] = ("announcement", "pre_auction")
) -> dict:
    """For each of the 5 real inputs and each cutoff: append one
    synthetic, far-future-dated row to ONLY that input, rebuild, and
    prove no pre-existing historical auction's predictor values change.
    For the auction-sample perturbation, a genuinely new auction key
    appears, so only the pre-existing keys are compared; for the 4
    external-source perturbations, no new auction is added, so the
    *entire* matrix (same key set) must remain bit-identical -- a
    future release must never be selected by any historical cutoff.
    """
    baselines = {c: build_matrix_for_tables(tables, cutoff_name=c, entries=entries) for c in cutoff_names}
    results: dict[str, dict] = {}
    for source_name in PERTURBABLE_SOURCES:
        extended_tables = _with_future_row_appended(tables, source_name=source_name)
        per_cutoff: dict[str, dict] = {}
        for cutoff_name in cutoff_names:
            result = build_matrix_for_tables(extended_tables, cutoff_name=cutoff_name, entries=entries)
            baseline = baselines[cutoff_name]
            if source_name == "sample":
                baseline_indexed = baseline.set_index(AUCTION_KEY_COL)
                result_indexed = result.set_index(AUCTION_KEY_COL)
                try:
                    pd.testing.assert_frame_equal(baseline_indexed, result_indexed.loc[baseline_indexed.index])
                    per_cutoff[cutoff_name] = {"identical_for_historical_rows": True, "n_historical_rows": len(baseline)}
                except AssertionError as exc:
                    per_cutoff[cutoff_name] = {"identical_for_historical_rows": False, "error": str(exc)}
            else:
                try:
                    pd.testing.assert_frame_equal(baseline, result)
                    per_cutoff[cutoff_name] = {"identical_for_historical_rows": True, "n_historical_rows": len(baseline)}
                except AssertionError as exc:
                    per_cutoff[cutoff_name] = {"identical_for_historical_rows": False, "error": str(exc)}
        results[source_name] = per_cutoff
    return results
