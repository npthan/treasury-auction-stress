"""Phase 7 report rendering -- every table below is computed directly
from the actual out-of-sample prediction tables
(`run_evaluation_phase7.Phase7Results`), never hand-typed, mirroring
`treasury_auction_stress.evaluation.reporting`'s discipline exactly.
"""

from __future__ import annotations

import pandas as pd

from treasury_auction_stress.evaluation.metrics import (
    build_breakdown_metrics_table,
    build_paired_model_vs_benchmark_by_year,
    build_provisional_only_metrics_table,
    compute_error_metrics,
    r2_vs_fold_own_mean,
)
from treasury_auction_stress.evaluation.probabilistic_metrics import (
    NOMINAL_INTERVALS,
    compute_coverage_table,
    compute_pinball_table,
    summarize_probabilistic_by,
)
from treasury_auction_stress.evaluation.reporting import df_to_markdown
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    POINT_MODEL_IDS,
    PROBABILISTIC_MODEL_IDS,
)
from treasury_auction_stress.evaluation.run_evaluation_phase7 import (
    TARGET_COL as TARGET_NAME,
)
from treasury_auction_stress.evaluation.stress_event import (
    MIN_POOLED_POSITIVE_COUNT,
    MIN_YEAR_POSITIVE_COUNT,
    compute_calibration_table,
    compute_classifier_metrics,
)

CUTOFF_VIEWS = ("announcement", "pre_auction")
FROZEN_BASELINE = "baseline_recent_history_frozen"
ADAPTIVE_BASELINE = "baseline_recent_history_adaptive"
CHALLENGERS = ("challenger_gbm_core", "challenger_shrinkage_tenor_reopening")
# Acceptance-review note: FROZEN_BASELINE/ADAPTIVE_BASELINE/CHALLENGERS
# split POINT_MODEL_IDS into the named subsets this report needs; their
# consistency with POINT_MODEL_IDS (the single source of truth) is
# checked directly by tests/test_phase7_protocol_contract.py, never
# assumed.


# ============================================================
# Point-forecast tables and report
# ============================================================


def build_point_pooled_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled point-forecast metrics, complete test years only. Unlike
    Phase 6's `metrics.build_pooled_metrics_table`, this does NOT
    compute `r2_vs_strong_baseline` against Phase 6's
    `baseline_tenor_reopening_mean` -- that model does not exist in
    Phase 7's own model list, and the relevant benchmark comparisons
    here are the paired recent-history tables below, not a benchmark-
    relative R2 against a different, Phase-6-only strong baseline.
    """
    rows = []
    complete = predictions.loc[~predictions["is_provisional"]]
    for cutoff_view in CUTOFF_VIEWS:
        for model_id in POINT_MODEL_IDS:
            subset = complete.loc[(complete["cutoff_view"] == cutoff_view) & (complete["model_id"] == model_id)]
            metrics = compute_error_metrics(subset, target_name=TARGET_NAME)
            metrics["r2_vs_fold_own_mean"] = r2_vs_fold_own_mean(subset)
            rows.append({"cutoff_view": cutoff_view, "model_id": model_id, **metrics})
    return pd.DataFrame(rows)


def build_point_provisional_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_provisional_only_metrics_table(
        predictions, target_names=(TARGET_NAME,), model_ids=POINT_MODEL_IDS, cutoff_views=CUTOFF_VIEWS
    )


def build_point_by_year_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_breakdown_metrics_table(
        predictions,
        target_names=(TARGET_NAME,),
        model_ids=POINT_MODEL_IDS,
        cutoff_views=CUTOFF_VIEWS,
        group_col="test_year",
        complete_years_only=False,
    )


def build_point_by_tenor_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_breakdown_metrics_table(
        predictions,
        target_names=(TARGET_NAME,),
        model_ids=POINT_MODEL_IDS,
        cutoff_views=CUTOFF_VIEWS,
        group_col="tenor",
        complete_years_only=True,
    )


def build_paired_vs_baseline_table(predictions: pd.DataFrame, *, model_ids: tuple, benchmark_model_id: str) -> pd.DataFrame:
    rows = []
    for cutoff_view in CUTOFF_VIEWS:
        table = build_paired_model_vs_benchmark_by_year(
            predictions, target_name=TARGET_NAME, cutoff_view=cutoff_view, model_ids=model_ids, benchmark_model_id=benchmark_model_id
        )
        table.insert(0, "cutoff_view", cutoff_view)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def apply_decision_rule(paired_vs_frozen: pd.DataFrame, paired_vs_adaptive: pd.DataFrame) -> dict:
    """`configs/phase_7_protocol.yml`'s `decision_rule_for_reported_results`:
    a challenger is called an "improvement" only if it beats BOTH
    baselines on pooled MAE AND per-year win count, at BOTH cutoff
    views. Anything less is "mixed."
    """
    verdicts: dict = {}
    for model_id in CHALLENGERS:
        model_verdicts = []
        for cutoff_view in CUTOFF_VIEWS:
            for label, paired in (("frozen", paired_vs_frozen), ("adaptive", paired_vs_adaptive)):
                complete = paired.loc[
                    (paired["model_id"] == model_id) & (paired["cutoff_view"] == cutoff_view) & (~paired["is_provisional"])
                ]
                if complete.empty:
                    continue
                pooled_mae_model = (complete["mae_model"] * complete["n"]).sum() / complete["n"].sum()
                pooled_mae_benchmark = (complete["mae_benchmark"] * complete["n"]).sum() / complete["n"].sum()
                years_model_better = int((complete["n_model_better"] > complete["n_benchmark_better"]).sum())
                years_total = len(complete)
                beats_pooled = pooled_mae_model < pooled_mae_benchmark
                beats_year_count = years_model_better > (years_total - years_model_better)
                model_verdicts.append(
                    {
                        "cutoff_view": cutoff_view,
                        "vs": label,
                        "beats_pooled_mae": beats_pooled,
                        "beats_per_year_count": beats_year_count,
                        "pooled_mae_model": pooled_mae_model,
                        "pooled_mae_benchmark": pooled_mae_benchmark,
                    }
                )
        overall = all(v["beats_pooled_mae"] and v["beats_per_year_count"] for v in model_verdicts) and bool(model_verdicts)
        verdicts[model_id] = {"detail": model_verdicts, "meets_bar": overall}
    return verdicts


def render_point_results_report(
    *,
    predictions: pd.DataFrame,
    pooled: pd.DataFrame,
    provisional: pd.DataFrame,
    by_year: pd.DataFrame,
    by_tenor: pd.DataFrame,
    paired_vs_frozen: pd.DataFrame,
    paired_vs_adaptive: pd.DataFrame,
    paired_adaptive_vs_frozen: pd.DataFrame,
    decision_rule_verdicts: dict,
    predictions_digest: str,
) -> str:
    verdict_lines = []
    for model_id, info in decision_rule_verdicts.items():
        verdict_lines.append(f"- `{model_id}`: **{'MEETS the bar' if info['meets_bar'] else 'mixed / does not meet the bar'}** (beats both recent-history baselines on pooled MAE and per-year win count, at both cutoffs, required for a 'win' claim)")

    return f"""# Phase 7 Point-Forecast Results

Generated by `treasury_auction_stress.evaluation.phase7_cli` directly
from `data/processed/phase_7_point_predictions.parquet` -- every number
below is a real out-of-sample metric from this session's own run.
Predictions content digest: `{predictions_digest}`.

**This is an auction-mechanics research result, not a trading
strategy.** No Sharpe ratio or P&L is reported anywhere in this
project.

## Decision-rule verdicts (`configs/phase_7_protocol.yml`)

{chr(10).join(verdict_lines)}

A challenger only counts as a "win" if it beats BOTH
`baseline_recent_history_frozen` AND `baseline_recent_history_adaptive`
on pooled MAE AND on a per-year win-count basis, at BOTH cutoff views.
Meeting the bar at only one cutoff, or on pooled MAE while losing the
per-year count, is reported as mixed -- never rounded up to a win.

## Pooled metrics, complete test years (2015-2025)

{df_to_markdown(pooled)}

## Provisional partial year (2026), reported separately -- never pooled into the complete-year numbers above

{df_to_markdown(provisional)}

## By test year

{df_to_markdown(by_year)}

## By tenor, complete years only

{df_to_markdown(by_tenor)}

## Paired comparison: `baseline_recent_history_adaptive` vs. `baseline_recent_history_frozen`

The adaptive benchmark is allowed to use an already-resolved earlier
test-year auction's own result (once safely available by the current
auction's own cutoff); the frozen benchmark is Phase 6's original,
unchanged for the whole test year. Reported separately, per the task
specification -- never presented as a win over each other by default.

{df_to_markdown(paired_adaptive_vs_frozen)}

## Paired comparison: each challenger vs. `baseline_recent_history_frozen`

{df_to_markdown(paired_vs_frozen.loc[paired_vs_frozen["model_id"].isin(CHALLENGERS)])}

## Paired comparison: each challenger vs. `baseline_recent_history_adaptive`

{df_to_markdown(paired_vs_adaptive.loc[paired_vs_adaptive["model_id"].isin(CHALLENGERS)])}

## The 2020-2021 regime-shift years

The `by_year` table above includes 2020 and 2021 as ordinary rows --
inspect them directly there. This project's own prior-phase finding
(`docs/target_specification.md`) is that primary-dealer share underwent
a persistent, large, one-directional level shift beginning in 2020; any
backward-looking baseline (frozen or adaptive) necessarily lags a
persistent shift by construction, and a fitted model with contemporaneous
predictors (the GBM challenger) may or may not track it better -- read
directly off the numbers above rather than assumed.

## Limitations

- Every point-forecast comparison here uses complete test years
  2015-2025 plus 2026 reported separately as provisional, identical to
  Phase 6's own fold schedule -- these years were already inspected in
  Phase 6, so this is a restrained, pre-declared comparison (the
  protocol was written to the working tree before scoring, but remains
  untracked by git as of this report -- not a claim of git-verifiable
  pre-registration), not a pristine first look.
- `challenger_gbm_core`'s missing-value handling (native, no imputer)
  differs from the linear models' median-imputation scheme -- both are
  trained strictly within their own fold, but the two are not
  mechanically identical pipelines.
- No feature ablation or architecture search beyond the single
  pre-declared GBM configuration was run.
"""


# ============================================================
# Probabilistic-forecast tables and report
# ============================================================


def build_probabilistic_pooled_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff_view in CUTOFF_VIEWS:
        for model_id in PROBABILISTIC_MODEL_IDS:
            for scope_label, include_provisional in (("complete_years", False), ("provisional_2026", True)):
                subset = predictions.loc[
                    (predictions["cutoff_view"] == cutoff_view) & (predictions["model_id"] == model_id)
                ]
                subset = subset.loc[subset["is_provisional"]] if include_provisional else subset.loc[~subset["is_provisional"]]
                quantile_levels = tuple(float(c[1:]) for c in subset.columns if c.startswith("q0."))
                pinball = compute_pinball_table(subset, quantile_levels=quantile_levels, target_name=TARGET_NAME)
                coverage = compute_coverage_table(subset, target_name=TARGET_NAME)
                rows.append(
                    {
                        "cutoff_view": cutoff_view,
                        "model_id": model_id,
                        "scope": scope_label,
                        **{k: v for k, v in pinball.items() if k in ("n", "pinball_mean", "unit")},
                        **{k: v for k, v in coverage.items() if k.startswith(("coverage_", "width_"))},
                    }
                )
    return pd.DataFrame(rows)


def build_probabilistic_by_year_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff_view in CUTOFF_VIEWS:
        for model_id in PROBABILISTIC_MODEL_IDS:
            subset = predictions.loc[
                (predictions["cutoff_view"] == cutoff_view) & (predictions["model_id"] == model_id) & (~predictions["is_provisional"])
            ]
            if subset.empty:
                continue
            quantile_levels = tuple(float(c[1:]) for c in subset.columns if c.startswith("q0."))
            table = summarize_probabilistic_by(subset, quantile_levels=quantile_levels, target_name=TARGET_NAME, group_col="test_year")
            table.insert(0, "model_id", model_id)
            table.insert(0, "cutoff_view", cutoff_view)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_probabilistic_by_tenor_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff_view in CUTOFF_VIEWS:
        for model_id in PROBABILISTIC_MODEL_IDS:
            subset = predictions.loc[
                (predictions["cutoff_view"] == cutoff_view) & (predictions["model_id"] == model_id) & (~predictions["is_provisional"])
            ]
            if subset.empty:
                continue
            quantile_levels = tuple(float(c[1:]) for c in subset.columns if c.startswith("q0."))
            table = summarize_probabilistic_by(subset, quantile_levels=quantile_levels, target_name=TARGET_NAME, group_col="tenor")
            table.insert(0, "model_id", model_id)
            table.insert(0, "cutoff_view", cutoff_view)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_coverage_trend_summary(by_year: pd.DataFrame, *, interval_label: str = "90%", n_years_each_end: int = 3) -> pd.DataFrame:
    """Acceptance-review addition: pooled coverage over an entire
    backtest can average out to near-nominal while individual years
    swing from under- to over-coverage -- this compares the mean
    coverage over the FIRST `n_years_each_end` complete test years
    against the LAST `n_years_each_end`, per (model, cutoff_view), so
    that drift is visible directly rather than obscured by pooling.
    """
    col = f"coverage_{interval_label}"
    rows = []
    for (cutoff_view, model_id), group in by_year.groupby(["cutoff_view", "model_id"]):
        ordered = group.sort_values("test_year")
        if len(ordered) < 2 * n_years_each_end:
            continue
        early = ordered.iloc[:n_years_each_end]
        late = ordered.iloc[-n_years_each_end:]
        rows.append(
            {
                "cutoff_view": cutoff_view,
                "model_id": model_id,
                "early_years": f"{early['test_year'].min()}-{early['test_year'].max()}",
                "early_mean_coverage": float(early[col].mean()),
                "late_years": f"{late['test_year'].min()}-{late['test_year'].max()}",
                "late_mean_coverage": float(late[col].mean()),
                "drift": float(late[col].mean() - early[col].mean()),
            }
        )
    return pd.DataFrame(rows)


def render_probabilistic_results_report(
    *, pooled: pd.DataFrame, by_year: pd.DataFrame, by_tenor: pd.DataFrame, predictions_digest: str
) -> str:
    coverage_trend = build_coverage_trend_summary(by_year)
    return f"""# Phase 7 Probabilistic Forecast Results

Generated by `treasury_auction_stress.evaluation.phase7_cli` directly
from `data/processed/phase_7_probabilistic_predictions.parquet`.
Predictions content digest: `{predictions_digest}`.

Quantile levels: `{{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}}`. Nominal
intervals reported: {", ".join(label for label, *_ in NOMINAL_INTERVALS)}.
Calibration window: the fold's own full expanding-window training
pool, via out-of-fold walk-forward residuals -- see
`configs/phase_7_protocol.yml`'s `probabilistic` section. Both methods
here quantify OUTCOME uncertainty (residual/quantile dispersion); NEITHER
separately models PARAMETER uncertainty (how uncertain the estimated
group means or tree splits themselves are) -- disclosed explicitly, not
silently ignored.

## Pooled pinball loss and interval coverage/width

Complete test years (2015-2025) and the provisional 2026 year, reported
as separate rows -- never pooled together.

{df_to_markdown(pooled)}

## By test year

{df_to_markdown(by_year)}

## By tenor, complete years only

{df_to_markdown(by_tenor)}

## Early-vs-late 90% interval coverage drift (acceptance-review addition)

**Pooled coverage across an entire 11-year backtest can average out to
near-nominal while individual years swing from under- to
over-coverage.** The table below compares each model's mean 90%-interval
coverage over its first 3 complete test years against its last 3 --
computed directly from the by-year table above, never hand-typed:

{df_to_markdown(coverage_trend)}

A positive `drift` means coverage rose from the early years to the late
years (i.e. the interval under-covered early and over-covered late, or
both) -- this is NOT proof the interval is unreliable (the underlying
target's own volatility genuinely changed across 2015-2025, including
the documented 2020-2021 regime shift), but it IS proof that a single
pooled coverage number, by itself, is not sufficient evidence of
reliable calibration across the whole period -- consistent with why
this project reports coverage broken out by year in the first place.

## Reading interval coverage

A well-calibrated 80% interval should contain the actual outcome
roughly 80% of the time. Coverage is reported with the row count it was
computed over everywhere it appears -- a coverage fraction with no
denominator is not reported. No claim is made that nominal coverage
holds under a genuine regime shift (e.g. 2020-2021) -- this is a
training-only calibration, never recalibrated using the year being
scored. See the early-vs-late drift table above for direct evidence of
this: pooled coverage alone is not sufficient to claim reliable
calibration.
"""


# ============================================================
# Stress-event gate report
# ============================================================


def render_stress_gate_report(*, gate_decisions: dict, classifier_predictions: pd.DataFrame, years_excluded: dict) -> str:
    sections = []
    for cutoff_view, decision in gate_decisions.items():
        sections.append(f"### Cutoff view: `{cutoff_view}`\n")
        sections.append(
            f"- **Gate decision: {'PASSES' if decision.passes else 'FAILS'}**\n"
            f"- Pooled positive-label count (complete years): {decision.pooled_positive_count} "
            f"(minimum required: {MIN_POOLED_POSITIVE_COUNT})\n"
            f"- Pooled labeled count (complete years): {decision.pooled_labeled_count}\n"
            f"- Missing training labels across all folds (warm-up years, excluded, never coded as negative): "
            f"{decision.n_missing_training_labels}\n"
            f"- Per-year positive counts: {decision.per_year_positive_counts}\n"
            f"- Years with reportable per-year metrics (>= {MIN_YEAR_POSITIVE_COUNT} positives): "
            f"{decision.years_with_reportable_metrics}\n"
            f"- Years suppressed (< {MIN_YEAR_POSITIVE_COUNT} positives, per-year metrics not shown): "
            f"{decision.years_suppressed}\n"
        )
        if decision.reasons:
            sections.append("Reasons:\n" + "\n".join(f"- {r}" for r in decision.reasons) + "\n")

        excluded = years_excluded.get(cutoff_view, {})
        if excluded:
            sections.append(
                "Folds excluded from the classifier entirely (cross-fitted training label had fewer than "
                f"2 distinct classes): {list(excluded.keys())}\n"
            )

        if not decision.passes:
            sections.append(
                "**No classifier score is published for this cutoff view.** Per the task specification, this "
                "is a documented blocker, not a fabricated score.\n"
            )
            continue

        clf = classifier_predictions.loc[classifier_predictions["cutoff_view"] == cutoff_view] if not classifier_predictions.empty else pd.DataFrame()
        complete = clf.loc[~clf["is_provisional"]] if not clf.empty else clf
        pooled_metrics = compute_classifier_metrics(complete) if not complete.empty else {}
        sections.append(f"**Pooled classifier metrics (complete years):** {pooled_metrics}\n")
        if pooled_metrics.get("no_skill_brier_score") is not None:
            verdict = "POORLY CALIBRATED" if pooled_metrics["worse_than_no_skill"] else "better calibrated than the no-skill baseline"
            sections.append(
                f"**Calibration verdict:** the classifier's Brier score ({pooled_metrics['brier_score']:.5f}) is "
                f"{'WORSE' if pooled_metrics['worse_than_no_skill'] else 'better'} than a trivial no-skill forecast "
                f"that always predicts each fold's own training-label prevalence "
                f"(no-skill Brier {pooled_metrics['no_skill_brier_score']:.5f}) -- **{verdict}**, despite a PR-AUC of "
                f"{pooled_metrics['pr_auc']:.5f} against a {pooled_metrics['prevalence'] * 100:.1f}% base rate "
                f"({pooled_metrics['n_positive']} positive of {pooled_metrics['n']} labeled auctions, "
                f"{pooled_metrics.get('confusion_false_positive', 'n/a')} false positives at the decision rule). "
                "A non-trivial ranking ability (PR-AUC above the base rate) does not imply well-calibrated "
                "PROBABILITIES -- both are reported here precisely so neither is read in isolation.\n"
            )

        calibration = compute_calibration_table(complete) if not complete.empty else pd.DataFrame()
        sections.append("**Calibration table (predicted-probability quintile vs. observed frequency):**\n")
        sections.append(df_to_markdown(calibration) + "\n")

        per_year_rows = []
        for year in decision.years_with_reportable_metrics:
            year_df = complete.loc[complete["test_year"] == year] if not complete.empty else pd.DataFrame()
            if year_df.empty:
                continue
            metrics = compute_classifier_metrics(year_df)
            metrics["test_year"] = year
            per_year_rows.append(metrics)
        sections.append("**Per-year classifier metrics (years with >= minimum positive count only):**\n")
        sections.append(df_to_markdown(pd.DataFrame(per_year_rows)) + "\n")

        provisional = clf.loc[clf["is_provisional"]] if not clf.empty else pd.DataFrame()
        if not provisional.empty:
            prov_metrics = compute_classifier_metrics(provisional)
            sections.append(f"**Provisional 2026 year, reported separately:** {prov_metrics}\n")

    return f"""# Phase 7 Dealer Absorption Surprise / Stress-Event Gate

Generated by `treasury_auction_stress.evaluation.phase7_cli`. The
stress-event label is an OPERATIONAL high-dealer-absorption flag
(the 90th percentile of a cross-fitted Dealer Absorption Surprise
series) -- it is NOT a claim that a flagged auction "failed" or that
the market was "distressed."

{chr(10).join(sections)}
"""


# ============================================================
# Leakage audit report
# ============================================================

PHASE7_LEAKAGE_TEST_FILES: tuple[tuple[str, str], ...] = (
    (
        "tests/test_phase7_safe_as_of_lookback.py",
        (
            "the shared safe-as-of-trailing-mean primitive: same-day/same-tenor exclusion, neighboring "
            "not-yet-available results, publication-date equality at the cutoff, missing values, sparse "
            "(20-Year) history, shuffled-row invariance, future-row poisoning, year boundaries, duplicate-"
            "column self-protection"
        ),
    ),
    ("tests/test_phase7_adaptive_baseline.py", "the adaptive recent-history benchmark updates within a test year, never uses a same-day sibling, falls back correctly for an unseen tenor"),
    ("tests/test_phase7_walk_forward.py", "generic walk-forward cross-fitting: excludes the earliest year, never uses a year's own future rows to fit its own model, respects the availability rule at year boundaries"),
    ("tests/test_phase7_gbm.py", "GBM point/quantile challenger: inner-CV vs. fallback hyperparameters, unseen-category handling, all-missing-column drop, quantile-crossing fix"),
    ("tests/test_phase7_shrinkage.py", "empirical Bayes shrinkage estimator: near-zero/near-full shrinkage limits, sparse-vs-populated group weighting, single-tenor degenerate case"),
    ("tests/test_phase7_probabilistic_metrics.py", "pinball loss, coverage, and width computed against hand-derived values; empty-frame handling"),
    ("tests/test_phase7_quantile_residual.py", "walk-forward residual quantile calibration: sparse-tenor fallback, single-year degenerate case, offsets centered on the supplied point forecast"),
    ("tests/test_phase7_logistic_classifier.py", "stress classifier: inner-CV vs. fallback C, missing-label exclusion, fewer-than-2-classes rejection"),
    ("tests/test_phase7_stress_event.py", "safe regime feature, cross-fitted training labels never coerced to False, threshold never fit on a year's own future outlier, gate pass/fail/suppression logic, classifier metrics and calibration table"),
    ("tests/test_phase7_protocol.py", "the frozen Phase 7 protocol loads and its fold schedule matches Phase 6's exactly"),
    ("tests/test_phase7_evaluation.py", "real-data integration: unique keys per model/cutoff, finite clipped predictions, identical auction-key coverage across all models and both cutoffs, non-crossing quantiles, gate-count consistency, no forbidden column in the core tier"),
    ("tests/test_phase7_leakage_adversarial.py", "cross-cutting adversarial cases: quantile calibration never influenced by a poisoned future year, pre-auction cutoff view enforces the same availability boundary, missing offering_amt never silently coerced"),
    ("tests/test_phase7_reporting.py", "every Phase 7 report-rendering function runs end-to-end on a real (small) result set"),
    ("tests/test_phase7_plots.py", "figure-generation smoke tests"),
    (
        "tests/test_phase7_cli.py",
        (
            "fail-before-write CLI discipline (deliberately-failing gate leaves prior outputs untouched, "
            "atomic writes) -- run by the ordinary `uv run pytest` suite, deliberately NOT included in this "
            "in-process gate itself (it contains tests that invoke phase7_cli.run(...), which would "
            "otherwise re-trigger this same gate recursively)"
        ),
    ),
)


def render_phase7_leakage_audit_report(*, pytest_summary: str, gate_decisions: dict) -> str:
    rows = "\n".join(f"| `{path}` | {covers} |" for path, covers in PHASE7_LEAKAGE_TEST_FILES)
    gate_lines = "\n".join(
        f"- `{cutoff_view}`: {'PASSES' if d.passes else 'FAILS'} ({d.pooled_positive_count} pooled positives, "
        f"{d.n_missing_training_labels} missing training labels excluded, never coded negative)"
        for cutoff_view, d in gate_decisions.items()
    )
    return f"""# Phase 7 Leakage Audit

Generated by `treasury_auction_stress.evaluation.phase7_cli`, which
actually re-runs the Phase 7 test suite as part of producing this
report -- the pass/fail counts below are from that real run, not a
claim.

## Stress-event gate outcome (re-stated here; full detail in `phase_7_stress_event_gate.md`)

{gate_lines}

## Every Phase 7 leakage-relevant test file and what it covers

| file | covers |
|---|---|
{rows}

## Actual test run output (this session)

```
{pytest_summary}
```

## Reproduce

```bash
uv run python -m treasury_auction_stress.evaluation.phase7_cli
uv run pytest -q tests/test_phase7_*.py
```
"""
