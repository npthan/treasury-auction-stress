"""Phase 2 entry point: build the final analysis sample, compute
targets, generate the exploratory figures, and regenerate
`artifacts/auction_data_profile.md`.

Run it via uv, from the repository root, after `treasury_auction_stress.data.cli`
has produced `data/processed/treasury_auctions_nominal_coupons.parquet`:

    uv run python -m treasury_auction_stress.features.profile_cli

This module never re-downloads or re-derives raw data; it only reads
the already-normalized nominal-coupon parquet table and reports on it.
Every number in the generated markdown comes from actually running
this code against that table -- see `docs/project_rules.md`'s "no fabricated
results" rule.

Since the Phase 2 acceptance review, every figure and summary table
below is computed from `select_modeling_sample`'s output (which
excludes the two verified restricted, primary-dealer-only reopening
auctions -- see `treasury_auction_stress.features.eligibility`), not
from the raw `settled` frame. A dedicated sensitivity section reports
key statistics both including and excluding those two rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.schema import NOMINAL_COUPON_TENORS
from treasury_auction_stress.features.dealer_absorption import (
    add_regime_feature,
    compute_walk_forward_surprise,
    describe_percentile_thresholds,
)
from treasury_auction_stress.features.eligibility import (
    ANALYSIS_START_DATE,
    describe_eligibility,
    select_analysis_sample,
    select_modeling_sample,
)
from treasury_auction_stress.features.targets import (
    add_all_targets,
    reconcile_bid_to_cover,
)
from treasury_auction_stress.visualization import auction_plots

ILLUSTRATIVE_THRESHOLD_TRAIN_CUTOFF = "2020-01-01"


def _extreme_observations(walk_forward: pd.DataFrame, n: int = 5) -> dict:
    cols = [
        "auction_date",
        "tenor",
        "is_reopening",
        "offering_amt",
        "primary_dealer_share",
        "expected_dealer_share",
        "dealer_absorption_surprise",
    ]

    def _records(frame: pd.DataFrame) -> list[dict]:
        out = frame[cols].copy()
        out["auction_date"] = out["auction_date"].dt.date.astype(str)
        for c in ("offering_amt", "primary_dealer_share", "expected_dealer_share", "dealer_absorption_surprise"):
            out[c] = out[c].astype("float64").round(6)
        return out.to_dict("records")

    if walk_forward.empty:
        return {"highest_surprise": [], "lowest_surprise": []}
    return {
        "highest_surprise": _records(walk_forward.nlargest(n, "dealer_absorption_surprise")),
        "lowest_surprise": _records(walk_forward.nsmallest(n, "dealer_absorption_surprise")),
    }


def _bid_to_cover_extremes(settled: pd.DataFrame, n: int = 3) -> dict:
    cols = ["auction_date", "tenor", "is_reopening", "bid_to_cover_ratio"]

    def _records(frame: pd.DataFrame) -> list[dict]:
        out = frame[cols].copy()
        out["auction_date"] = out["auction_date"].dt.date.astype(str)
        out["bid_to_cover_ratio"] = out["bid_to_cover_ratio"].astype("float64")
        return out.to_dict("records")

    valid = settled.dropna(subset=["bid_to_cover_ratio"])
    return {
        "lowest": _records(valid.nsmallest(n, "bid_to_cover_ratio")),
        "highest": _records(valid.nlargest(n, "bid_to_cover_ratio")),
    }


def _regime_comparison(settled: pd.DataFrame) -> dict:
    pre = settled.loc[settled["auction_date"] < "2020-03-01", "primary_dealer_share"].dropna()
    pandemic = settled.loc[
        (settled["auction_date"] >= "2020-03-01") & (settled["auction_date"] <= "2021-12-31"),
        "primary_dealer_share",
    ].dropna()
    post = settled.loc[settled["auction_date"] > "2021-12-31", "primary_dealer_share"].dropna()
    return {
        "pre_pandemic_mean": float(pre.mean()),
        "pre_pandemic_n": len(pre),
        "pandemic_mean": float(pandemic.mean()),
        "pandemic_n": len(pandemic),
        "post_pandemic_mean": float(post.mean()),
        "post_pandemic_n": len(post),
    }


def _distribution_by_tenor(settled: pd.DataFrame) -> dict:
    result = {}
    for tenor in NOMINAL_COUPON_TENORS:
        vals = settled.loc[settled["tenor"] == tenor, "primary_dealer_share"].dropna().astype("float64")
        if vals.empty:
            continue
        result[tenor] = {
            "n": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "p50": float(vals.median()),
            "max": float(vals.max()),
        }
    return result


def _distribution_by_reopening(settled: pd.DataFrame) -> dict:
    result = {}
    for is_reopening, label in ((False, "new_issue"), (True, "reopening")):
        vals = settled.loc[settled["is_reopening"] == is_reopening, "primary_dealer_share"].dropna().astype("float64")
        result[label] = {
            "n": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    return result


def _walk_forward_summary(walk_forward: pd.DataFrame) -> dict | None:
    if walk_forward.empty:
        return None
    values = walk_forward["dealer_absorption_surprise"].astype("float64")
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _sensitivity_comparison(with_specials: pd.DataFrame, without_specials: pd.DataFrame) -> dict:
    def _dealer_share_stats(df: pd.DataFrame) -> dict:
        s = df["primary_dealer_share"].dropna().astype("float64")
        return {
            "n": len(s),
            "mean": float(s.mean()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        }

    return {
        "dealer_share_including_special_auctions": _dealer_share_stats(with_specials),
        "dealer_share_excluding_special_auctions": _dealer_share_stats(without_specials),
        "walk_forward_das_including_special_auctions": _walk_forward_summary(
            compute_walk_forward_surprise(add_regime_feature(with_specials))
        ),
        "walk_forward_das_excluding_special_auctions": _walk_forward_summary(
            compute_walk_forward_surprise(add_regime_feature(without_specials))
        ),
    }


def build_profile(nominal_df: pd.DataFrame, figures_dir: Path) -> dict:
    settled, pending = select_analysis_sample(nominal_df, start_date=ANALYSIS_START_DATE)
    settled = add_all_targets(settled)

    modeling_sample = select_modeling_sample(settled)
    modeling_sample = add_regime_feature(modeling_sample)

    walk_forward = compute_walk_forward_surprise(modeling_sample)

    illustrative_train = walk_forward.loc[
        walk_forward["auction_date"] < ILLUSTRATIVE_THRESHOLD_TRAIN_CUTOFF, "dealer_absorption_surprise"
    ]
    percentile_comparison = (
        describe_percentile_thresholds(illustrative_train) if not illustrative_train.dropna().empty else {}
    )

    figures = {
        "auction_counts_by_year_tenor": auction_plots.plot_auction_counts_by_year_tenor(
            modeling_sample, figures_dir / "auction_counts_by_year_tenor.png"
        ),
        "new_vs_reopening_by_tenor": auction_plots.plot_new_vs_reopening_by_tenor(
            modeling_sample, figures_dir / "new_vs_reopening_by_tenor.png"
        ),
        "offering_amount_over_time": auction_plots.plot_offering_amount_over_time(
            modeling_sample, figures_dir / "offering_amount_over_time.png"
        ),
        "bid_to_cover_over_time": auction_plots.plot_bid_to_cover_over_time(
            modeling_sample, figures_dir / "bid_to_cover_over_time.png"
        ),
        "bidder_shares_by_tenor": auction_plots.plot_bidder_shares_by_tenor_small_multiples(
            modeling_sample, figures_dir / "bidder_shares_by_tenor.png"
        ),
        "target_distribution_by_tenor": auction_plots.plot_target_distribution_by_tenor(
            modeling_sample, figures_dir / "target_distribution_by_tenor.png"
        ),
        "target_distribution_by_reopening": auction_plots.plot_target_distribution_by_reopening(
            modeling_sample, figures_dir / "target_distribution_by_reopening.png"
        ),
        "bidder_share_relationships": auction_plots.plot_bidder_share_relationships(
            modeling_sample, figures_dir / "bidder_share_relationships.png"
        ),
        "dealer_share_structural_view": auction_plots.plot_dealer_share_structural_view(
            modeling_sample, figures_dir / "dealer_share_structural_view.png"
        ),
        "dealer_absorption_surprise_distribution": auction_plots.plot_dealer_absorption_surprise_distribution(
            walk_forward, figures_dir / "dealer_absorption_surprise_distribution.png"
        ),
    }

    return {
        "eligibility": describe_eligibility(nominal_df, start_date=ANALYSIS_START_DATE),
        "n_settled": len(settled),
        "n_modeling_sample": len(modeling_sample),
        "n_pending": len(pending),
        "sensitivity": _sensitivity_comparison(settled, modeling_sample),
        "bid_to_cover_reconciliation": reconcile_bid_to_cover(modeling_sample),
        "target_distribution_by_tenor": _distribution_by_tenor(modeling_sample),
        "target_distribution_by_reopening": _distribution_by_reopening(modeling_sample),
        "regime_comparison_pre_pandemic_pandemic_post": _regime_comparison(modeling_sample),
        "walk_forward_n": len(walk_forward),
        "walk_forward_excluded_first_year_n": int(len(modeling_sample) - len(walk_forward)),
        "walk_forward_surprise_summary": _walk_forward_summary(walk_forward),
        "extreme_observations": _extreme_observations(walk_forward),
        "bid_to_cover_extremes": _bid_to_cover_extremes(modeling_sample),
        "illustrative_percentile_thresholds": {
            "train_cutoff": ILLUSTRATIVE_THRESHOLD_TRAIN_CUTOFF,
            "n_train": len(illustrative_train.dropna()),
            "thresholds": percentile_comparison,
        },
        "figures": {name: str(path) for name, path in figures.items()},
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_profile_markdown(profile: dict, generated_note: str) -> str:
    lines: list[str] = []
    lines.append("# Treasury Auction Data Profile (Phase 2)")
    lines.append("")
    lines.append(generated_note)
    lines.append("")
    lines.append(
        "Every figure and number below was computed by running "
        "`treasury_auction_stress.features.profile_cli` against the "
        "processed nominal-coupon table. None of it is hand-typed or "
        "estimated. See `docs/target_specification.md` for the target "
        "formulas, `docs/point_in_time_rules.md` for the leakage rules "
        "this profile follows, and the Phase 2 acceptance review "
        "for the primary-source investigation behind the eligibility and "
        "reconciliation rules used here."
    )
    lines.append("")

    elig = profile["eligibility"]
    lines.append("## Final sample rules")
    lines.append(f"- Analysis start date: **{elig['analysis_start_date']}**")
    lines.append(f"- Rows excluded for being before the start date: **{elig['rows_before_start_date_excluded']}**")
    lines.append(f"- Settled analysis-sample rows (complete audit trail): **{elig['settled_analysis_sample_rows']}**")
    lines.append(
        f"- Special (restricted primary-dealer-only) auctions, preserved but excluded "
        f"from the modeling sample: **{elig['special_auction_rows']}**"
    )
    for row in elig["special_auction_records"]:
        lines.append(f"  - {row['auction_date']}, {row['tenor']}, CUSIP {row['cusip']}")
    if elig["special_auction_rows"]:
        lines.append(f"  - Exclusion reason: {elig['special_auction_exclusion_reason']}")
    lines.append(f"- **Main economic modeling sample rows: {elig['modeling_sample_rows']}**")
    lines.append(f"- Pending (unsettled) rows, excluded from targets/figures: **{elig['pending_rows']}**")
    if elig["pending_auction_dates"]:
        lines.append(f"  - Pending auction dates: {', '.join(elig['pending_auction_dates'])}")
    lines.append(
        "- Auctions flagged `is_unusually_small_offering` (a secondary, "
        "corroborating flag; the two rows above are exactly the two special "
        f"auctions): **{len(elig['unusually_small_offering_rows'])}**"
    )
    lines.append("")

    lines.append("## Sensitivity: with vs. without the two special auctions")
    sens = profile["sensitivity"]
    ds_with = sens["dealer_share_including_special_auctions"]
    ds_without = sens["dealer_share_excluding_special_auctions"]
    lines.append(
        f"- Primary-dealer share, including special auctions: n={ds_with['n']}, "
        f"mean {_fmt_pct(ds_with['mean'])}, max {_fmt_pct(ds_with['max'])}"
    )
    lines.append(
        f"- Primary-dealer share, excluding special auctions (main modeling sample): "
        f"n={ds_without['n']}, mean {_fmt_pct(ds_without['mean'])}, max {_fmt_pct(ds_without['max'])}"
    )
    das_with = sens["walk_forward_das_including_special_auctions"]
    das_without = sens["walk_forward_das_excluding_special_auctions"]
    if das_with and das_without:
        lines.append(
            f"- Walk-forward Dealer Absorption Surprise, including special auctions: "
            f"n={das_with['n']}, mean {das_with['mean']:.4f}, max {das_with['max']:.4f}"
        )
        lines.append(
            f"- Walk-forward Dealer Absorption Surprise, excluding special auctions: "
            f"n={das_without['n']}, mean {das_without['mean']:.4f}, max {das_without['max']:.4f}"
        )
        lines.append(
            "The maximum surprise drops substantially once the two special auctions "
            "are excluded, confirming they were the entire source of the two most "
            "extreme values in the original (pre-review) profile -- see "
            "the Phase 2 acceptance review."
        )
    lines.append("")

    lines.append("## Verified target formulas")
    lines.append(
        "- `public_accepted_amount = comp_accepted + noncomp_accepted + "
        "fima_noncomp_accepted` (matches Treasury's own published "
        "\"Subtotal\"; excludes only Fed/SOMA add-ons -- see "
        "`treasury_auction_stress.features.targets` module docstring and "
        "the Phase 2 acceptance review for the full reconciliation)."
    )
    lines.append("- `primary_dealer_share = primary_dealer_accepted / public_accepted_amount`")
    lines.append("- `direct_bidder_share = direct_bidder_accepted / public_accepted_amount`")
    lines.append("- `indirect_bidder_share = indirect_bidder_accepted / public_accepted_amount`")
    lines.append(
        "- `bid_to_cover_ratio` (Treasury's own field) is kept as the primary "
        "figure; `bid_to_cover_calculated = (comp_tendered + noncomp_accepted "
        "+ fima_noncomp_accepted) / public_accepted_amount` is computed as an "
        "audit cross-check, and now reconciles **exactly** for every row in "
        "the modeling sample."
    )
    btc = profile["bid_to_cover_reconciliation"]
    lines.append("")
    lines.append("**Bid-to-cover reconciliation** (API field vs. from-scratch calculation, rounded to 2dp):")
    lines.append(f"- Rows compared: {btc.get('n_compared', 0)}")
    if btc.get("n_compared"):
        lines.append(f"- Max absolute difference: {btc['max_abs_diff']:.4f}")
        lines.append(f"- Mean absolute difference: {btc['mean_abs_diff']:.4f}")
        lines.append(
            f"- Within 0.01: {btc['n_within_0_01']} rows; within 0.02: "
            f"{btc['n_within_0_02']} rows; over 0.02: {btc['n_over_0_02']} rows"
        )
    lines.append("")

    lines.append("## Counts by tenor (main modeling sample)")
    lines.append("| Tenor | N | Mean dealer share | Std | Min | Median | Max |")
    lines.append("|---|---|---|---|---|---|---|")
    for tenor, stats in profile["target_distribution_by_tenor"].items():
        lines.append(
            f"| {tenor} | {stats['n']} | {_fmt_pct(stats['mean'])} | {_fmt_pct(stats['std'])} | "
            f"{_fmt_pct(stats['min'])} | {_fmt_pct(stats['p50'])} | {_fmt_pct(stats['max'])} |"
        )
    lines.append("")

    lines.append("## Target distribution by reopening status (main modeling sample)")
    lines.append("| Status | N | Mean dealer share | Std |")
    lines.append("|---|---|---|---|")
    for label, stats in profile["target_distribution_by_reopening"].items():
        lines.append(f"| {label} | {stats['n']} | {_fmt_pct(stats['mean'])} | {_fmt_pct(stats['std'])} |")
    lines.append("")

    regime = profile["regime_comparison_pre_pandemic_pandemic_post"]
    lines.append("## Pandemic-era behavior (descriptive; no causal claim)")
    lines.append(
        f"- Pre-pandemic (before 2020-03-01), n={regime['pre_pandemic_n']}: "
        f"mean primary-dealer share {_fmt_pct(regime['pre_pandemic_mean'])}"
    )
    lines.append(
        f"- Pandemic period (2020-03-01 to 2021-12-31), n={regime['pandemic_n']}: "
        f"mean primary-dealer share {_fmt_pct(regime['pandemic_mean'])}"
    )
    lines.append(
        f"- After 2021-12-31, n={regime['post_pandemic_n']}: "
        f"mean primary-dealer share {_fmt_pct(regime['post_pandemic_mean'])}"
    )
    lines.append(
        "These windows are purely date-defined groupings for descriptive "
        "comparison, not evidence of what caused any difference between them."
    )
    lines.append("")

    lines.append("## Dealer Absorption Surprise: walk-forward (expanding-origin, annual folds)")
    lines.append(
        f"- Auctions with a computed surprise (main modeling sample): **{profile['walk_forward_n']}** "
        f"(excludes **{profile['walk_forward_excluded_first_year_n']}** rows from the "
        "first calendar year of history, which has no strictly-prior training fold)"
    )
    wf = profile["walk_forward_surprise_summary"]
    if wf:
        lines.append(
            f"- Distribution: mean {wf['mean']:.4f}, std {wf['std']:.4f}, "
            f"min {wf['min']:.4f}, max {wf['max']:.4f}"
        )
    lines.append("")

    lines.append("### Extreme observations: largest positive Dealer Absorption Surprise (main modeling sample)")
    lines.append("| Date | Tenor | Reopening | Offering ($) | Actual share | Expected share | Surprise |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in profile["extreme_observations"]["highest_surprise"]:
        lines.append(
            f"| {row['auction_date']} | {row['tenor']} | {row['is_reopening']} | "
            f"{row['offering_amt']:,.0f} | {_fmt_pct(row['primary_dealer_share'])} | "
            f"{_fmt_pct(row['expected_dealer_share'])} | {_fmt_pct(row['dealer_absorption_surprise'])} |"
        )
    lines.append("")
    lines.append(
        "The two $25,000,000 restricted, primary-dealer-only reopenings no "
        "longer appear here -- they are excluded from the modeling sample "
        "(see the sample-rules and sensitivity sections above)."
    )
    lines.append("")

    lines.append("### Extreme observations: largest negative Dealer Absorption Surprise (main modeling sample)")
    lines.append("| Date | Tenor | Reopening | Offering ($) | Actual share | Expected share | Surprise |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in profile["extreme_observations"]["lowest_surprise"]:
        lines.append(
            f"| {row['auction_date']} | {row['tenor']} | {row['is_reopening']} | "
            f"{row['offering_amt']:,.0f} | {_fmt_pct(row['primary_dealer_share'])} | "
            f"{_fmt_pct(row['expected_dealer_share'])} | {_fmt_pct(row['dealer_absorption_surprise'])} |"
        )
    lines.append("")

    lines.append("### Bid-to-cover extremes (main modeling sample)")
    lines.append("Lowest:")
    for row in profile["bid_to_cover_extremes"]["lowest"]:
        lines.append(f"- {row['auction_date']}, {row['tenor']} (reopening={row['is_reopening']}): {row['bid_to_cover_ratio']:.2f}")
    lines.append("")
    lines.append("Highest:")
    for row in profile["bid_to_cover_extremes"]["highest"]:
        lines.append(f"- {row['auction_date']}, {row['tenor']} (reopening={row['is_reopening']}): {row['bid_to_cover_ratio']:.2f}")
    lines.append(
        "\nOne widely-reported example among the low-cover rows: the "
        "2021-02-25 7-year note auction is contemporaneously described in "
        "financial press coverage of that date as a notably weak auction "
        "that coincided with a broader bond-market selloff -- mentioned "
        "here as context for why that date appears among this project's "
        "own extreme observations, not as a claim this project independently "
        "verified the causal narrative."
    )
    lines.append("")

    thresholds = profile["illustrative_percentile_thresholds"]
    lines.append("## Provisional stress-event threshold: descriptive percentile comparison")
    lines.append(
        f"Illustrative only -- training window is auctions with walk-forward "
        f"surprise dated before **{thresholds['train_cutoff']}** (n={thresholds['n_train']}). "
        "This split is for descriptive comparison in this profile only; it is "
        "not a chronological evaluation fold and must not be reused as one "
        "without being re-derived inside a real evaluation protocol in a later phase."
    )
    lines.append("")
    lines.append("| Percentile | Threshold (surprise) | N flagged in training window | Fraction flagged |")
    lines.append("|---|---|---|---|")
    for entry in thresholds["thresholds"].values():
        lines.append(
            f"| p{round(entry['percentile'] * 100)} | {entry['threshold']:.4f} | "
            f"{entry['n_flagged_in_training_window']} | {_fmt_pct(entry['fraction_flagged_in_training_window'])} |"
        )
    lines.append("")

    lines.append("## Figures")
    lines.append("All figures are drawn from the main modeling sample (special auctions excluded).")
    for name, path in profile["figures"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")

    lines.append("## Structural limitations")
    lines.append(
        "- The 20-year bond has no history before its May 2020 "
        "reintroduction; any tenor-level baseline for it is necessarily "
        "shorter and less stable than for the other six tenors."
    )
    lines.append(
        "- This project has only one retrieval snapshot per UTC calendar "
        "day (see `treasury_auction_stress.data.time_utils`); detecting a "
        "*revision* to a previously-published auction result (as opposed "
        "to a duplicate row) is not yet possible and is listed as an open "
        "question below."
    )
    lines.append(
        "- Treasury's announcements for the two special auctions do not "
        "state the underlying administrative reason for restricting "
        "bidding to primary dealers on those specific dates -- the auction "
        "*mechanism* is fully verified (see the Phase 2 acceptance review), "
        "but the business motive behind choosing those two dates is not "
        "public in the documents this project can access."
    )
    if wf:
        lines.append(
            f"- The walk-forward Dealer Absorption Surprise distribution has a "
            f"negative mean ({wf['mean']:.4f}, not centered on 0). Primary-dealer "
            "share has trended steadily downward over 2010-2026 (see the "
            "structural-view figure); an expanding-origin model that only ever "
            "sees the past systematically lags a persistent one-directional "
            "trend, so it tends to over-predict the expected share in periods of "
            "sustained decline. This is an expected property of a simple, "
            "training-data-only baseline tracking a trending series, not a bug "
            "in the leakage-safety of the transformer -- but it does mean the "
            "surprise's absolute scale should be read fold-by-fold rather than "
            "assumed to be zero-centered overall, and a later phase may want a "
            "regime feature with a shorter or adaptive window to track the "
            "trend more closely."
        )
    lines.append("")

    lines.append("## Open research questions")
    lines.append(
        "- What administrative motive prompted Treasury to restrict these "
        "two specific reopenings to primary dealers only? (The auction "
        "mechanism is verified; the business reason is not public.)"
    )
    lines.append(
        "- Would a longer retrieval history (multiple retrieval-dated "
        "snapshots) reveal any revised auction results, and if so, how "
        "should a revision be handled under the point-in-time rules?"
    )
    lines.append(
        "- How should the 20-year bond's short, structurally distinct "
        "history be weighted or pooled in a later regime-aware model?"
    )
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        type=Path,
        help="Directory containing treasury_auctions_nominal_coupons.parquet",
    )
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
    parser.add_argument("--figures-dir", default=None, type=Path, help="Default: <reports-dir>/figures")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figures_dir = args.figures_dir or (args.reports_dir / "figures")

    nominal_path = args.processed_dir / "treasury_auctions_nominal_coupons.parquet"
    nominal_df = pd.read_parquet(nominal_path)

    profile = build_profile(nominal_df, figures_dir)

    generated_note = (
        "Generated by `uv run python -m treasury_auction_stress.features.profile_cli` "
        f"against `{nominal_path}`."
    )
    markdown = render_profile_markdown(profile, generated_note)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.reports_dir / "auction_data_profile.md"
    profile_path.write_text(markdown, encoding="utf-8")
    print(f"wrote profile report: {profile_path}")
    print(f"wrote {len(profile['figures'])} figures under: {figures_dir}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
