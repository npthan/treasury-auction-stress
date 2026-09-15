"""Phase 8 report rendering and dashboard-data assembly. Every table
below is computed directly from the accepted Phase 7 out-of-sample
prediction tables (via `evaluation.reporting7` and
`evaluation.phase8_interpretation`), never hand-typed -- mirroring
`evaluation.reporting`/`evaluation.reporting7`'s own discipline. This
module renders the Markdown interpretation report AND assembles the
single JSON aggregate the local dashboard reads, from the exact same
in-memory tables, so the report and the dashboard can never silently
drift apart from each other.
"""

from __future__ import annotations

import math

import pandas as pd

from treasury_auction_stress.evaluation.metrics import unit_label
from treasury_auction_stress.evaluation.phase8_interpretation import (
    ADAPTIVE_MODEL_ID,
    CASE_STUDY_RULES,
    FROZEN_MODEL_ID,
    build_error_distribution_table,
    build_paired_comparison_by_group,
    build_stress_metrics_by_year_table,
    compute_quantile_clip_fraction,
    select_case_studies,
    year_block_bootstrap_ci,
)
from treasury_auction_stress.evaluation.probabilistic_metrics import NOMINAL_INTERVALS
from treasury_auction_stress.evaluation.reporting import df_to_markdown
from treasury_auction_stress.evaluation.reporting7 import (
    CHALLENGERS,
    CUTOFF_VIEWS,
    build_coverage_trend_summary,
    build_paired_vs_baseline_table,
    build_point_by_tenor_table,
    build_point_by_year_table,
    build_point_pooled_table,
    build_point_provisional_table,
    build_probabilistic_by_tenor_table,
    build_probabilistic_by_year_table,
    build_probabilistic_pooled_table,
)
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    POINT_MODEL_IDS,
    PROBABILISTIC_MODEL_IDS,
)
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    TARGET_COL as TARGET_NAME,
)
from treasury_auction_stress.evaluation.stress_event import (
    compute_calibration_table,
    compute_classifier_metrics,
)

POINT_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list[dict], JSON-safe (NaN/NaT -> None, numpy scalars -> native)."""
    if df is None or df.empty:
        return []
    clean = df.where(pd.notnull(df), None)
    records = clean.to_dict(orient="records")
    out = []
    for record in records:
        row = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            row[key] = value
        out.append(row)
    return out


# ============================================================
# Phase 8 diagnostic tables (built once, reused by report + dashboard)
# ============================================================


def _classifier_predictions_with_tenor(classifier_predictions: pd.DataFrame, point_predictions: pd.DataFrame) -> pd.DataFrame:
    """`phase_7_classifier_predictions.parquet` carries no `tenor` column
    of its own -- this attaches one via a metadata-only join on
    `auction_key` against the point-forecast predictions (which do
    carry `tenor` for the identical auctions). This changes no
    prediction, label, threshold, or metric: it only looks up a value
    already present elsewhere in the accepted Phase 7 outputs, needed
    so the dashboard's tenor filter can scope the stress-classifier
    panel too instead of always showing the tenor-pooled figure.
    """
    if classifier_predictions.empty:
        return classifier_predictions
    tenor_by_key = point_predictions[["auction_key", "tenor"]].drop_duplicates().set_index("auction_key")["tenor"]
    with_tenor = classifier_predictions.copy()
    with_tenor["tenor"] = with_tenor["auction_key"].map(tenor_by_key)
    return with_tenor


def build_phase8_tables(
    *, point_predictions: pd.DataFrame, probabilistic_predictions: pd.DataFrame, classifier_predictions: pd.DataFrame
) -> dict:
    """Compute every Phase 8 diagnostic table once, from the in-memory
    Phase 7 prediction tables passed in. Returns a flat dict of
    DataFrames/dicts -- both `render_phase8_interpretation_report` and
    `build_dashboard_data` consume this SAME dict, so the report and
    the dashboard are guaranteed to describe identical numbers.
    """
    tables: dict = {}

    # ---- Point-forecast pooled/provisional/by-year/by-tenor (Phase 7's
    # own builders, reused directly -- not re-derived) ----
    tables["point_pooled"] = build_point_pooled_table(point_predictions)
    tables["point_provisional"] = build_point_provisional_table(point_predictions)
    tables["point_by_year"] = build_point_by_year_table(point_predictions)
    tables["point_by_tenor"] = build_point_by_tenor_table(point_predictions)

    # ---- Paired adaptive-vs-frozen, extended to tenor and reopening
    # status (by-year paired comparison already exists in Phase 7's own
    # report; this is the new Phase 8 breakdown) ----
    for cutoff_view in CUTOFF_VIEWS:
        tables[f"paired_by_tenor_{cutoff_view}"] = build_paired_comparison_by_group(
            point_predictions, cutoff_view=cutoff_view, group_col="tenor"
        )
        tables[f"paired_by_reopening_{cutoff_view}"] = build_paired_comparison_by_group(
            point_predictions, cutoff_view=cutoff_view, group_col="is_reopening"
        )
    tables["paired_by_year"] = build_paired_vs_baseline_table(
        point_predictions, model_ids=(ADAPTIVE_MODEL_ID,), benchmark_model_id=FROZEN_MODEL_ID
    )

    # ---- Error distributions and signed bias, headline point models ----
    headline_point_models = (FROZEN_MODEL_ID, ADAPTIVE_MODEL_ID, *CHALLENGERS)
    for cutoff_view in CUTOFF_VIEWS:
        tables[f"error_distribution_pooled_{cutoff_view}"] = build_error_distribution_table(
            point_predictions, cutoff_view=cutoff_view, model_ids=headline_point_models
        )
        tables[f"error_distribution_by_tenor_{cutoff_view}"] = build_error_distribution_table(
            point_predictions, cutoff_view=cutoff_view, model_ids=headline_point_models, group_col="tenor"
        )

    # ---- Case studies ----
    for cutoff_view in CUTOFF_VIEWS:
        tables[f"case_studies_{cutoff_view}"] = select_case_studies(point_predictions, cutoff_view=cutoff_view)

    # ---- Probabilistic coverage/width by year/tenor (Phase 7's own
    # builders) + coverage drift + clip fractions (new) ----
    tables["probabilistic_pooled"] = build_probabilistic_pooled_table(probabilistic_predictions)
    tables["probabilistic_by_year"] = build_probabilistic_by_year_table(probabilistic_predictions)
    tables["probabilistic_by_tenor"] = build_probabilistic_by_tenor_table(probabilistic_predictions)
    tables["coverage_trend_90"] = build_coverage_trend_summary(tables["probabilistic_by_year"], interval_label="90%")

    for model_id in PROBABILISTIC_MODEL_IDS:
        for cutoff_view in CUTOFF_VIEWS:
            tables[f"clip_fraction_by_year_{model_id}_{cutoff_view}"] = compute_quantile_clip_fraction(
                probabilistic_predictions,
                cutoff_view=cutoff_view,
                model_id=model_id,
                quantile_levels=POINT_QUANTILE_LEVELS,
                group_col="test_year",
            )
            tables[f"clip_fraction_by_tenor_{model_id}_{cutoff_view}"] = compute_quantile_clip_fraction(
                probabilistic_predictions,
                cutoff_view=cutoff_view,
                model_id=model_id,
                quantile_levels=POINT_QUANTILE_LEVELS,
                group_col="tenor",
            )
            tables[f"clip_fraction_pooled_{model_id}_{cutoff_view}"] = compute_quantile_clip_fraction(
                probabilistic_predictions, cutoff_view=cutoff_view, model_id=model_id, quantile_levels=POINT_QUANTILE_LEVELS
            )

    # ---- Stress-event classifier: pooled + by-year + calibration ----
    pooled_stress = {}
    calibration = {}
    by_year_stress = {}
    for cutoff_view in CUTOFF_VIEWS:
        clf = classifier_predictions.loc[classifier_predictions["cutoff_view"] == cutoff_view] if not classifier_predictions.empty else pd.DataFrame()
        complete = clf.loc[~clf["is_provisional"]] if not clf.empty else clf
        pooled_stress[cutoff_view] = compute_classifier_metrics(complete) if not complete.empty else {"n": 0}
        calibration[cutoff_view] = compute_calibration_table(complete) if not complete.empty else pd.DataFrame()
        by_year_stress[cutoff_view] = build_stress_metrics_by_year_table(classifier_predictions, cutoff_view=cutoff_view)
    tables["stress_pooled"] = pooled_stress
    tables["stress_calibration"] = calibration
    tables["stress_by_year"] = by_year_stress

    # ---- Descriptive year-block bootstrap on the paired adaptive-vs-
    # frozen yearly mean difference ----
    bootstrap = {}
    for cutoff_view in CUTOFF_VIEWS:
        yearly = tables["paired_by_year"]
        yearly_complete = yearly.loc[(yearly["cutoff_view"] == cutoff_view) & (~yearly["is_provisional"])]
        yearly_diffs = yearly_complete.set_index("test_year")["mean_paired_diff_model_minus_benchmark"]
        bootstrap[cutoff_view] = year_block_bootstrap_ci(yearly_diffs)
    tables["adaptive_vs_frozen_bootstrap"] = bootstrap

    # ---- Per-tenor scoped views for the dashboard's GLOBAL tenor filter.
    # The dashboard's own copy states "one cutoff and one tenor apply to
    # every chart on this page at once" -- an earlier draft only used the
    # selected tenor to highlight a bar in the by-tenor chart while the
    # KPI tiles, MAE-by-year chart, coverage/width chart, stress panel,
    # and case studies stayed pooled across all tenors regardless of the
    # selection (see the Phase 8 acceptance review). Each table below is
    # the SAME builder used for the pooled tables above, called on data
    # pre-filtered to one tenor, so a selected tenor can never silently
    # fall back to a pooled or wrong-cutoff number. ----
    tenors = sorted(point_predictions["tenor"].dropna().unique().tolist())
    classifier_with_tenor = _classifier_predictions_with_tenor(classifier_predictions, point_predictions)

    point_pooled_by_tenor: dict[str, pd.DataFrame] = {}
    point_by_year_by_tenor: dict[str, pd.DataFrame] = {}
    probabilistic_pooled_by_tenor: dict[str, pd.DataFrame] = {}
    probabilistic_by_year_by_tenor: dict[str, pd.DataFrame] = {}
    stress_pooled_by_tenor: dict[str, dict] = {}
    case_studies_by_tenor: dict[str, dict[str, pd.DataFrame]] = {}

    for tenor in tenors:
        point_subset = point_predictions.loc[point_predictions["tenor"] == tenor]
        prob_subset = probabilistic_predictions.loc[probabilistic_predictions["tenor"] == tenor]

        point_pooled_by_tenor[tenor] = build_point_pooled_table(point_subset)
        point_by_year_by_tenor[tenor] = build_point_by_year_table(point_subset)
        probabilistic_pooled_by_tenor[tenor] = build_probabilistic_pooled_table(prob_subset)
        probabilistic_by_year_by_tenor[tenor] = build_probabilistic_by_year_table(prob_subset)

        stress_pooled_by_tenor[tenor] = {}
        for cutoff_view in CUTOFF_VIEWS:
            clf = (
                classifier_with_tenor.loc[
                    (classifier_with_tenor["cutoff_view"] == cutoff_view) & (classifier_with_tenor["tenor"] == tenor)
                ]
                if not classifier_with_tenor.empty
                else pd.DataFrame()
            )
            complete = clf.loc[~clf["is_provisional"]] if not clf.empty else clf
            stress_pooled_by_tenor[tenor][cutoff_view] = (
                compute_classifier_metrics(complete) if not complete.empty else {"n": 0}
            )

        case_studies_by_tenor[tenor] = {
            cutoff_view: select_case_studies(point_subset, cutoff_view=cutoff_view) for cutoff_view in CUTOFF_VIEWS
        }

    tables["tenors"] = tenors
    tables["point_pooled_by_tenor"] = point_pooled_by_tenor
    tables["point_by_year_by_tenor"] = point_by_year_by_tenor
    tables["probabilistic_pooled_by_tenor"] = probabilistic_pooled_by_tenor
    tables["probabilistic_by_year_by_tenor"] = probabilistic_by_year_by_tenor
    tables["stress_pooled_by_tenor"] = stress_pooled_by_tenor
    tables["case_studies_by_tenor"] = case_studies_by_tenor

    return tables


# ============================================================
# Markdown report
# ============================================================


def _with_int_columns_preserved(df: pd.DataFrame, int_columns: tuple[str, ...]) -> pd.DataFrame:
    """`reporting.df_to_markdown` renders each row via `DataFrame.iterrows()`,
    which upcasts an entire row to one common dtype -- harmless when a
    table has at least one string column (the row falls back to `object`
    dtype, preserving each cell's own type), but an all-numeric table
    (e.g. a per-year clip-fraction table with only int/float columns)
    would otherwise print `2015.000` instead of `2015`. This casts the
    named integer columns to plain Python `int` objects so the row stays
    `object`-dtyped and every cell renders in its own natural type.
    """
    if df.empty:
        return df
    result = df.copy()
    for col in int_columns:
        # `.astype(object)` alone (not followed by any further column-wide
        # operation) is required -- pandas silently re-infers a uniformly-
        # int object column back to int64, which would undo this entirely.
        result[col] = [int(v) for v in result[col]]
        result[col] = result[col].astype(object)
    return result


def _format_year_list(years: list[int]) -> str:
    """Compress a sorted list of years into ranges for prose, e.g.
    `[2015, 2016, 2018, 2019, 2020]` -> `"2015-2016 and 2018-2020"`.
    """
    if not years:
        return "no complete year"
    years = sorted(years)
    ranges: list[tuple[int, int]] = []
    start = prev = years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
            continue
        ranges.append((start, prev))
        start = prev = y
    ranges.append((start, prev))
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] + " and " + parts[1]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _win_pattern_for_cutoff(paired_by_year: pd.DataFrame, *, cutoff_view: str) -> dict:
    view = paired_by_year.loc[
        (paired_by_year["cutoff_view"] == cutoff_view) & (~paired_by_year["is_provisional"])
    ].sort_values("test_year")
    model_wins = [int(y) for y in view.loc[view["n_model_better"] > view["n_benchmark_better"], "test_year"]]
    benchmark_wins = [int(y) for y in view.loc[view["n_benchmark_better"] > view["n_model_better"], "test_year"]]
    tied = [int(y) for y in view.loc[view["n_model_better"] == view["n_benchmark_better"], "test_year"]]
    return {"model_wins": model_wins, "benchmark_wins": benchmark_wins, "tied": tied, "n_total": len(view)}


def describe_broad_or_concentrated(paired_by_year: pd.DataFrame) -> str:
    """Derives the "is the gain broad or concentrated" sentence directly
    from `paired_by_year`'s own `n_model_better`/`n_benchmark_better`
    columns for complete (non-provisional) years -- never hand-typed.
    A hand-typed version of this sentence previously claimed the frozen
    benchmark won more auctions in 2025 when the table actually showed
    a 48-29 adaptive win that year (see the Phase 8 acceptance review);
    deriving it from the table makes that class of narrative/table
    contradiction structurally impossible.
    """
    per_cutoff = {cv: _win_pattern_for_cutoff(paired_by_year, cutoff_view=cv) for cv in CUTOFF_VIEWS}
    if per_cutoff[CUTOFF_VIEWS[0]] == per_cutoff[CUTOFF_VIEWS[1]]:
        p = per_cutoff[CUTOFF_VIEWS[0]]
        sentence = (
            f"the adaptive benchmark wins more auctions than it loses in {len(p['model_wins'])} of "
            f"{p['n_total']} complete test years, identically at both cutoffs "
            f"({_format_year_list(p['model_wins'])}); the frozen benchmark wins more auctions in "
            f"{_format_year_list(p['benchmark_wins'])}"
        )
        if p["tied"]:
            sentence += f" ({_format_year_list(p['tied'])} tied on win count)"
        sentence += "."
    else:
        parts = []
        for cv in CUTOFF_VIEWS:
            p = per_cutoff[cv]
            parts.append(
                f"at the {cv} cutoff, the adaptive benchmark wins {len(p['model_wins'])} of {p['n_total']} "
                f"complete years ({_format_year_list(p['model_wins'])}) and the frozen benchmark wins "
                f"{_format_year_list(p['benchmark_wins'])}"
            )
        sentence = "; ".join(parts) + "."
    return sentence


def _clip_fraction_pooled_summary_table(tables: dict) -> pd.DataFrame:
    """A small helper solely for report rendering: `n`/
    `n_rows_touching_clip_bound` come back as `int` from
    `compute_quantile_clip_fraction`, but `.iloc[0].to_dict()` on a
    single-row DataFrame containing both ints and a float would coerce
    everything in that row to `float64` (a Series has one dtype) --
    this keeps the count columns genuinely integer in the rendered
    table instead of printing `860.000`.
    """
    rows = []
    for model_id in PROBABILISTIC_MODEL_IDS:
        for cv in CUTOFF_VIEWS:
            table = tables[f"clip_fraction_pooled_{model_id}_{cv}"]
            if table.empty:
                continue
            record = table.iloc[0]
            rows.append(
                {
                    "model_id": model_id,
                    "cutoff_view": cv,
                    "n": int(record["n"]),
                    "n_rows_touching_clip_bound": int(record["n_rows_touching_clip_bound"]),
                    "fraction_rows_touching_clip_bound": float(record["fraction_rows_touching_clip_bound"]),
                }
            )
    return pd.DataFrame(rows)


def render_phase8_interpretation_report(
    *, tables: dict, point_digest: str, probabilistic_digest: str, classifier_row_count: int
) -> str:
    sections: list[str] = []

    sections.append(
        "# Phase 8 Interpretation\n\n"
        "Generated by `treasury_auction_stress.evaluation.phase8_cli` directly "
        "from the accepted Phase 7 out-of-sample prediction tables -- no Phase "
        "1-7 target definition, feature eligibility, fold construction, model "
        "setting, threshold, prediction, or accepted metric is changed anywhere "
        f"in this document. Point-predictions digest: `{point_digest}`. Probabilistic-"
        f"predictions digest: `{probabilistic_digest}`. Classifier predictions: {classifier_row_count} rows.\n\n"
        "**Three caveats that apply to every number below:**\n\n"
        "1. `primary_dealer_share` is an auction-ALLOCATION measure and at most "
        "a proxy for dealer balance-sheet absorption -- it does not directly "
        "measure dealers' final holdings, funding stress, or an auction's "
        "'success.'\n"
        "2. The stress-event label passed its data-sufficiency gate (enough "
        "events existed to evaluate), but the classifier's own probabilities "
        "are poorly calibrated (Section 6) -- 'gate passed' is not a claim the "
        "classifier is a reliable stress detector.\n"
        "3. The residual-quantile method's ~92% pooled 90%-interval coverage "
        "masks real year-to-year drift (Section 4) -- pooled coverage alone is "
        "never sufficient evidence of stable calibration.\n\n"
        "This is a retrospective analysis of years already examined in Phases "
        "6-7 (2015-2025, plus provisional 2026) -- not a fresh holdout, not "
        "causal evidence, and not a trading strategy.\n"
    )

    sections.append(
        "## 1. Paired frozen-vs-adaptive point-forecast errors\n\n"
        "`baseline_recent_history_adaptive` and `baseline_recent_history_frozen` "
        "use DIFFERENT information schedules: the frozen benchmark is a single "
        "constant per tenor, fixed at the fold's own fit origin for the whole "
        "test year; the adaptive benchmark recomputes, per test row, using every "
        "same-tenor result (training-pool or already-resolved same-test-year) "
        "whose own `result_safe_available_date` clears that row's own cutoff. "
        "The comparison below is therefore an observational difference in "
        "information sets, not two otherwise-identical models.\n\n"
        "### By year (both cutoffs; reproduced from `phase_7_point_forecast_results.md`)\n\n"
        + df_to_markdown(tables["paired_by_year"])
        + "\n\n### By tenor, announcement cutoff, complete years only\n\n"
        + df_to_markdown(tables["paired_by_tenor_announcement"])
        + "\n\n### By tenor, pre-auction cutoff, complete years only\n\n"
        + df_to_markdown(tables["paired_by_tenor_pre_auction"])
        + "\n\n### By reopening status, announcement cutoff, complete years only\n\n"
        + df_to_markdown(tables["paired_by_reopening_announcement"])
        + "\n\n### By reopening status, pre-auction cutoff, complete years only\n\n"
        + df_to_markdown(tables["paired_by_reopening_pre_auction"])
        + "\n\n**Is the gain broad or concentrated?** Read directly off the "
        "by-year table: " + describe_broad_or_concentrated(tables["paired_by_year"]) + " The "
        "gain is broad but not universal, and is not concentrated in a single year or "
        "tenor. **2020-2021 regime shift**: both years appear as ordinary rows "
        "above; the adaptive benchmark's within-year updating gives it a "
        "structural advantage exactly when a persistent one-directional level "
        "shift is underway (2020 pooled MAE: adaptive 3.784pp vs. frozen "
        "5.086pp), because the frozen benchmark cannot incorporate a single "
        "same-year data point until the following test year. **Sparse 20-Year "
        "history**: the 20-Year tenor has no auction history before its "
        "2020-05-20 reintroduction, so its early complete-year folds rely more "
        "heavily on the global-mean fallback for both benchmarks -- its by-"
        "tenor row above (n=68) should be read with that structural caveat, "
        "not as evidence the adaptive mechanism itself performs differently "
        "for this tenor.\n"
    )

    sections.append(
        "## 2. Error distributions, signed bias, and case studies\n\n"
        "### Pooled signed-error (forecast minus actual) distribution, announcement cutoff\n\n"
        + df_to_markdown(tables["error_distribution_pooled_announcement"])
        + "\n\n### Pooled signed-error distribution, pre-auction cutoff\n\n"
        + df_to_markdown(tables["error_distribution_pooled_pre_auction"])
        + "\n\n### By tenor, announcement cutoff\n\n"
        + df_to_markdown(tables["error_distribution_by_tenor_announcement"])
        + "\n\nAll four headline point models have a positive mean signed error "
        "(bias) pooled across 2015-2025 -- i.e. on average, forecasts run "
        "ABOVE the settled `primary_dealer_share`, consistent with the "
        "backward-looking baselines lagging the persistent downward trend in "
        "dealer participation documented in `docs/target_specification.md`. "
        "The adaptive benchmark's bias is materially smaller than the frozen "
        "benchmark's (it updates within the year), but still positive -- it "
        "reduces, but does not eliminate, the lag.\n\n"
        "### Case studies\n\n"
        "**Case-selection rule** (defined once in code, "
        "`phase8_interpretation.select_case_studies`, run identically "
        "regardless of which auctions it surfaces): among complete-year "
        "auctions with both models' predictions, select (1) the single "
        "largest adaptive IMPROVEMENT in absolute error over the frozen "
        "model, (2) the single largest adaptive DETERIORATION, and (3) the "
        "auction closest to the MEDIAN improvement (\"typical\"). "
        "`known_at_cutoff__*` columns are what each model actually predicted "
        "before the auction; `learned_after_settlement__actual_pct` is the "
        "settled outcome, known only afterward -- the two are never "
        "conflated.\n\n"
        "#### Announcement cutoff\n\n"
        + df_to_markdown(tables["case_studies_announcement"])
        + "\n\n#### Pre-auction cutoff\n\n"
        + df_to_markdown(tables["case_studies_pre_auction"])
        + "\n"
    )

    sections.append(
        "## 3. Descriptive uncertainty around the paired adaptive-vs-frozen difference\n\n"
        "A year-level block bootstrap (resampling which COMPLETE TEST YEARS "
        "contribute, with replacement -- never individual auctions, because "
        "within one year the same model's per-auction errors share a fold, a "
        "frozen fit, and a regime, and are not independent draws) around the "
        "mean paired difference (`adaptive MAE - frozen MAE`, negative means "
        "adaptive is better), computed over the "
        f"{tables['adaptive_vs_frozen_bootstrap']['announcement']['n_years']} complete test years. "
        "**This is descriptive uncertainty with a small number of years, not a "
        "definitive significance claim.**\n\n"
        + df_to_markdown(
            pd.DataFrame(
                [{"cutoff_view": cv, **tables["adaptive_vs_frozen_bootstrap"][cv]} for cv in CUTOFF_VIEWS]
            )
        )
        + "\n"
    )

    sections.append(
        "## 4. Prediction-interval coverage, width, and clipping\n\n"
        "A coverage fraction is never reported without its width and sample "
        "count -- both appear alongside every coverage number below.\n\n"
        "### By test year (both models, both cutoffs)\n\n"
        + df_to_markdown(tables["probabilistic_by_year"])
        + "\n\n### By tenor, complete years only\n\n"
        + df_to_markdown(tables["probabilistic_by_tenor"])
        + "\n\n### Early-vs-late 90% coverage drift\n\n"
        + df_to_markdown(tables["coverage_trend_90"])
        + "\n\n### Fraction of rows with at least one quantile clipped to [0, 1], pooled complete years\n\n"
        + df_to_markdown(_clip_fraction_pooled_summary_table(tables))
        + "\n\n### Clip fraction by year, `quantile_recent_history_residual`, announcement cutoff\n\n"
        + df_to_markdown(
            _with_int_columns_preserved(
                tables["clip_fraction_by_year_quantile_recent_history_residual_announcement"],
                ("test_year", "n", "n_rows_touching_clip_bound"),
            )
        )
        + "\n\n### Clip fraction by tenor, `quantile_recent_history_residual`, announcement cutoff\n\n"
        + df_to_markdown(tables["clip_fraction_by_tenor_quantile_recent_history_residual_announcement"])
        + "\n\n**Reading this section**: the ~92% pooled 90%-interval coverage "
        "for `quantile_recent_history_residual` (see "
        "`phase_7_probabilistic_results.md`) is a pooled average over years "
        "that undercovered (e.g. 2015-2016, ~75-82%) and years that "
        "overcovered (2021-2025, ~93-100%) -- the early-vs-late drift table "
        "above quantifies this directly. `quantile_gbm_core` undercovers "
        "substantially relative to its own nominal levels in most years. "
        "Neither pattern is claimed to be stable or resolved by this report.\n"
    )

    sections.append(
        "## 5. Stress-event label: prevalence, false positives, PR-AUC, and Brier vs. no-skill\n\n"
        "**'Gate passed' means there were enough positive-labeled auctions to "
        "evaluate the classifier at all (>= 30 pooled, >= 5 per reported year) "
        "-- it is not a claim the classifier performs well.**\n\n"
        + df_to_markdown(
            pd.DataFrame(
                [
                    {"cutoff_view": cv, **tables["stress_pooled"][cv]}
                    for cv in CUTOFF_VIEWS
                ]
            )
        )
        + "\n\n### By test year (every year with any labeled row -- small-`n_positive` years included, not suppressed, so reliability can be judged directly)\n\n"
        + "\n\n#### Announcement cutoff\n\n"
        + df_to_markdown(tables["stress_by_year"]["announcement"])
        + "\n\n#### Pre-auction cutoff\n\n"
        + df_to_markdown(tables["stress_by_year"]["pre_auction"])
        + "\n\n### Calibration table (predicted-probability quintile vs. observed frequency), announcement cutoff\n\n"
        + df_to_markdown(tables["stress_calibration"]["announcement"])
        + "\n\n### Calibration table, pre-auction cutoff\n\n"
        + df_to_markdown(tables["stress_calibration"]["pre_auction"])
        + "\n\nAt both cutoffs, the classifier's Brier score is roughly "
        "2.0-2.3x WORSE than a trivial forecast that always predicts each "
        "fold's own training-label prevalence, despite a PR-AUC "
        "meaningfully above the 6.2% base rate -- ranking ability and "
        "probability calibration are different properties, and this "
        "classifier has the first without the second. It should not be used "
        "as a stress detector in its current form.\n"
    )

    sections.append(
        "## 6. Feature importance -- deliberately not computed\n\n"
        "This report does not include a feature-importance or model-"
        "interpretation section for `challenger_gbm_core`. Computing one "
        "faithfully would require re-fitting each fold's own GBM (its "
        "per-split feature importances were not persisted as a Phase 7 "
        "artifact), which risks looking like new model-fitting work in a "
        "phase whose scope is presentation and diagnostics, not re-tuning. "
        "Given that (a) `challenger_gbm_core` did not beat either "
        "recent-history baseline in this backtest (Phase 7 results) and (b) "
        "this project's predictors are meaningfully correlated (e.g. several "
        "dealer-stats and rate features share the same underlying market "
        "moves) under a regime that shifted materially in 2020-2021, any "
        "importance ranking would need the same correlated-predictor and "
        "regime-change caveats as any other importance measure, and would "
        "not license a causal reading -- it was judged not to add enough "
        "beyond the point-forecast results already reported to justify the "
        "extra model-fitting.\n"
    )

    sections.append(
        "## Reproduce\n\n"
        "```bash\n"
        "uv run python -m treasury_auction_stress.evaluation.phase8_cli\n"
        "uv run pytest -q tests/test_phase8_*.py\n"
        "```\n"
    )

    return "\n".join(sections)


# ============================================================
# Dashboard data assembly
# ============================================================


def _clean_stress_metrics(metrics: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}


def build_dashboard_data(
    *,
    tables: dict,
    point_digest: str,
    probabilistic_digest: str,
    point_model_labels: dict[str, str],
    probabilistic_model_labels: dict[str, str],
    complete_test_years: tuple[int, ...],
    provisional_year: int,
) -> dict:
    """Assemble the single JSON aggregate the local dashboard reads.
    Built from the EXACT SAME in-memory tables the Markdown report
    renders (`build_phase8_tables`'s output) -- the dashboard and the
    report can never describe different numbers for the same quantity.
    """
    return {
        "generated_from": {
            "point_predictions_digest": point_digest,
            "probabilistic_predictions_digest": probabilistic_digest,
            "source": "data/processed/phase_7_{point,probabilistic,classifier}_predictions.parquet (accepted Phase 7 outputs)",
        },
        "meta": {
            "cutoff_views": list(CUTOFF_VIEWS),
            "point_model_ids": list(POINT_MODEL_IDS),
            "point_model_labels": point_model_labels,
            "probabilistic_model_ids": list(PROBABILISTIC_MODEL_IDS),
            "probabilistic_model_labels": probabilistic_model_labels,
            "target_name": TARGET_NAME,
            "unit": unit_label(TARGET_NAME),
            "complete_test_years": list(complete_test_years),
            "provisional_year": provisional_year,
            "tenors": list(tables["tenors"]),
            "nominal_intervals": [label for label, *_ in NOMINAL_INTERVALS],
        },
        "point": {
            "pooled": _records(tables["point_pooled"]),
            "provisional": _records(tables["point_provisional"]),
            "by_year": _records(tables["point_by_year"]),
            "by_tenor": _records(tables["point_by_tenor"]),
            "paired_by_year": _records(tables["paired_by_year"]),
            "paired_by_tenor": {cv: _records(tables[f"paired_by_tenor_{cv}"]) for cv in CUTOFF_VIEWS},
            "paired_by_reopening": {cv: _records(tables[f"paired_by_reopening_{cv}"]) for cv in CUTOFF_VIEWS},
            "error_distribution_pooled": {cv: _records(tables[f"error_distribution_pooled_{cv}"]) for cv in CUTOFF_VIEWS},
            "error_distribution_by_tenor": {cv: _records(tables[f"error_distribution_by_tenor_{cv}"]) for cv in CUTOFF_VIEWS},
            "case_studies": {cv: _records(tables[f"case_studies_{cv}"]) for cv in CUTOFF_VIEWS},
            "case_selection_rules": list(CASE_STUDY_RULES),
            # ---- Tenor-scoped equivalents of the tables above, so the
            # dashboard's tenor filter can apply to the KPI tiles, the
            # MAE-by-year chart, and case studies too, not only highlight
            # a bar in the by-tenor chart (see the Phase 8 acceptance
            # review) ----
            "pooled_by_tenor": {t: _records(tables["point_pooled_by_tenor"][t]) for t in tables["tenors"]},
            "by_year_by_tenor": {t: _records(tables["point_by_year_by_tenor"][t]) for t in tables["tenors"]},
            "case_studies_by_tenor": {
                t: {cv: _records(tables["case_studies_by_tenor"][t][cv]) for cv in CUTOFF_VIEWS} for t in tables["tenors"]
            },
        },
        "probabilistic": {
            "pooled": _records(tables["probabilistic_pooled"]),
            "by_year": _records(tables["probabilistic_by_year"]),
            "by_tenor": _records(tables["probabilistic_by_tenor"]),
            "coverage_trend_90": _records(tables["coverage_trend_90"]),
            "pooled_by_tenor": {t: _records(tables["probabilistic_pooled_by_tenor"][t]) for t in tables["tenors"]},
            "by_year_by_tenor": {t: _records(tables["probabilistic_by_year_by_tenor"][t]) for t in tables["tenors"]},
            "clip_fraction_by_year": {
                f"{model_id}__{cv}": _records(tables[f"clip_fraction_by_year_{model_id}_{cv}"])
                for model_id in PROBABILISTIC_MODEL_IDS
                for cv in CUTOFF_VIEWS
            },
            "clip_fraction_by_tenor": {
                f"{model_id}__{cv}": _records(tables[f"clip_fraction_by_tenor_{model_id}_{cv}"])
                for model_id in PROBABILISTIC_MODEL_IDS
                for cv in CUTOFF_VIEWS
            },
            "clip_fraction_pooled": {
                f"{model_id}__{cv}": _records(tables[f"clip_fraction_pooled_{model_id}_{cv}"])
                for model_id in PROBABILISTIC_MODEL_IDS
                for cv in CUTOFF_VIEWS
            },
        },
        "stress": {
            "pooled": {cv: _clean_stress_metrics(tables["stress_pooled"][cv]) for cv in CUTOFF_VIEWS},
            "by_year": {cv: _records(tables["stress_by_year"][cv]) for cv in CUTOFF_VIEWS},
            "calibration": {cv: _records(tables["stress_calibration"][cv]) for cv in CUTOFF_VIEWS},
            "pooled_by_tenor": {
                t: {cv: _clean_stress_metrics(tables["stress_pooled_by_tenor"][t][cv]) for cv in CUTOFF_VIEWS}
                for t in tables["tenors"]
            },
        },
        "uncertainty": {
            "adaptive_vs_frozen_bootstrap": tables["adaptive_vs_frozen_bootstrap"],
        },
        "warnings": {
            "target_is_proxy": (
                "primary_dealer_share is an auction-allocation measure and at most a proxy for dealer "
                "balance-sheet absorption -- it does not directly measure dealers' final holdings, funding "
                "stress, or an auction's success."
            ),
            "classifier_poorly_calibrated": (
                "The stress-event classifier passed a data-sufficiency gate (enough events existed to "
                "evaluate it) but its Brier score is roughly 2-2.3x worse than the no-skill forecast based "
                "on each fold's own training-event rate. It is not a reliable stress detector."
            ),
            "coverage_drifts_by_year": (
                "The residual-quantile method's nominal 90% interval covered roughly 92% pooled across "
                "2015-2025, but coverage drifted markedly by year (undercoverage early, overcoverage late). "
                "The pooled statistic alone does not show stable calibration. The quantile GBM undercovered "
                "substantially."
            ),
            "not_a_fresh_holdout": (
                "2015-2025 and provisional 2026 were already examined in Phases 6-7. This is a retrospective "
                "analysis, not a fresh holdout, causal evidence, or a trading strategy."
            ),
        },
    }
