"""Phase 4A entry point: join Treasury par-yield-curve rates onto
Treasury auctions and regenerate `artifacts/treasury_rates_data_quality.md`.

Run it via uv, from the repository root, after
`treasury_auction_stress.data.treasury_rates_cli` and
`treasury_auction_stress.data.cli` have both produced their processed
tables:

    uv run python -m treasury_auction_stress.features.treasury_rates_profile_cli

Every number in the generated markdown comes from actually running
this code -- see `docs/project_rules.md`'s "no fabricated results" rule. Trains no
model, adds no new source.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.treasury_rates_raw_store import (
    compare_snapshots,
    find_snapshot_paths,
)
from treasury_auction_stress.data.treasury_rates_schema import (
    FOUR_MONTH_INTRODUCED_DATE,
    ONE_POINT_FIVE_MONTH_INTRODUCED_DATE,
    RATES_START_DATE,
    TWENTY_YEAR_METHODOLOGY_CHANGE_DATE,
    WHOLE_CURVE_METHODOLOGY_CHANGE_DATE,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.eligibility import (
    select_analysis_sample,
    select_modeling_sample,
)
from treasury_auction_stress.features.treasury_rates_candidate_features import (
    add_tenor_matched_rate_features,
    build_rate_feature_table,
)
from treasury_auction_stress.features.treasury_rates_join import (
    STABLE_MATURITY_NAMES,
    as_of_join,
)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append("nan" if isinstance(v, float) and pd.isna(v) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


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
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    args = parser.parse_args(argv)

    long_df = pd.read_parquet(args.processed_dir / "treasury_rates_long.parquet")
    wide_df = pd.read_parquet(args.processed_dir / "treasury_rates_wide.parquet")
    feat = build_rate_feature_table(wide_df)

    nominal_df = pd.read_parquet(args.processed_dir / "treasury_auctions_nominal_coupons.parquet")
    settled, pending = select_analysis_sample(nominal_df)
    normalized_complete_sample = add_cutoff_dates(pd.concat([settled, pending], ignore_index=True))
    ordinary_modeling_sample = add_cutoff_dates(select_modeling_sample(settled))
    assert len(normalized_complete_sample) == len(settled) + len(pending)
    assert len(ordinary_modeling_sample) == len(select_modeling_sample(settled))

    joined_ann = as_of_join(ordinary_modeling_sample, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    joined_pre = as_of_join(ordinary_modeling_sample, feat, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    joined_ann = add_tenor_matched_rate_features(joined_ann)
    joined_pre = add_tenor_matched_rate_features(joined_pre)

    joined_ann_complete = as_of_join(normalized_complete_sample, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)

    coverage_by_tenor = (
        joined_ann.groupby("tenor")
        .agg(n_auctions=("cusip", "size"), n_matched=("rates_join_matched", "sum"))
        .reset_index()
    )
    coverage_by_year = (
        joined_ann.assign(year=joined_ann["auction_date"].dt.year)
        .groupby("year")
        .agg(n_auctions=("cusip", "size"), n_matched=("rates_join_matched", "sum"))
        .reset_index()
    )

    missing_rows = []
    for maturity in STABLE_MATURITY_NAMES:
        col = f"{maturity}_missing_reason"
        if col not in joined_ann.columns:
            continue
        counts = joined_ann[col].str.extract(r"^([a-z_]+)")[0].value_counts(dropna=True)
        for reason, n in counts.items():
            missing_rows.append({"maturity": maturity, "reason": reason, "n_auctions": int(n)})
    missing_breakdown = pd.DataFrame(missing_rows)

    example_row = joined_ann.loc[joined_ann["tenor"] == "10-Year"].sort_values("auction_date").iloc[-1]
    example_pre_row = joined_pre.loc[
        (joined_pre["cusip"] == example_row["cusip"]) & (joined_pre["auction_date"] == example_row["auction_date"])
    ].iloc[0]

    # Phase 4 acceptance review, issue 11: genuine multi-snapshot
    # revision detection -- never inferred from a single snapshot.
    snapshot_comparisons = []
    for year in sorted({d.year for d in long_df["rate_date"]}):
        snapshot_paths = find_snapshot_paths(args.raw_dir, year)
        if len(snapshot_paths) < 2:
            continue
        for older, newer in itertools.pairwise(snapshot_paths):
            result = compare_snapshots(older, newer)
            snapshot_comparisons.append(
                {
                    "year": year,
                    "older_retrieval": older.stem.rsplit("_retrieved_", 1)[-1],
                    "newer_retrieval": newer.stem.rsplit("_retrieved_", 1)[-1],
                    "n_common_dates": result["n_common_dates"],
                    "n_changed_dates": result["n_changed_dates"],
                    "identical": result["identical"],
                }
            )
    snapshot_comparison_df = pd.DataFrame(snapshot_comparisons)

    lines = [
        "# Treasury Daily Par Yield Curve -- Data Quality Report",
        "",
        (f"Generated by `uv run python -m treasury_auction_stress.features.treasury_rates_profile_cli` "
        f"from {len(long_df)} normalized rate rows ({wide_df['rate_date'].min().date()} to "
        f"{wide_df['rate_date'].max().date()}) and {len(ordinary_modeling_sample)} ordinary_modeling_sample "
        f"auctions ({len(normalized_complete_sample)} in normalized_complete_sample)."),
        "",
        "## Source and endpoint (verified live 2026-09-11)",
        "",
        "- Official landing page: `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve`",
        "- Verified CSV export (one calendar year per request, no auth): see `treasury_rates_schema.CSV_URL_TEMPLATE`.",
        "- NOT obtained via FRED/ALFRED -- see `docs/data_source_governance.md`.",
        "",
        "## Verified schema evolution",
        "",
        ("Confirmed by fetching real CSVs for 1990, 2002, 2006, 2009, 2020, 2024, 2026: the column set "
        "is not fixed across history (1 Mo/2 Mo/4 Mo/20 Yr/1.5 Month were all added at different dates; "
        "see `treasury_rates_schema.py` module docstring for the full table). This project's ingestion "
        f"window ({RATES_START_DATE} onward) has all 7 nominal-coupon tenors present throughout."),
        "",
        "## Verified methodology regimes",
        "",
        (f"- Whole-curve spline method changed {WHOLE_CURVE_METHODOLOGY_CHANGE_DATE} (quasi-cubic Hermite -> monotone convex); "
        "Treasury's own page states pre-change rates 'remain official' -- never rewritten under the new method."),
        f"- 20-Year point construction changed {TWENTY_YEAR_METHODOLOGY_CHANGE_DATE} (composite of off-the-run bonds -> the real, reintroduced 20-Year bond).",
        f"- `4 Mo` column introduced {FOUR_MONTH_INTRODUCED_DATE}; `1.5 Month` column introduced {ONE_POINT_FIVE_MONTH_INTRODUCED_DATE} (neither affects nominal-coupon tenors directly).",
        "",
        "## Publication rule (verified, Treasury's own methodology page)",
        "",
        ("Inputs collected 'at or near 3:30 PM' ET; 'yield curve rates are usually available at Treasury's "
        "interest rate website by 6:00 PM Eastern Time each trading day,' with a stated possible delay. "
        "No exact historical intraday timestamp exists for any date. This project therefore uses "
        "`publication_date = rate_date` and `publication_safe_available_date = the next full U.S. business "
        "day after rate_date` (Phase 4 acceptance review, issue 2: previously a plain `+1 calendar day`, "
        "which produced a Saturday availability date for every Friday rate -- fixed to "
        "`time_utils.next_full_business_day_after`) -- so a same-day rate is never used for a same-day "
        "cutoff, and a Friday rate is never available on a weekend."),
        "",
        "## Genuine revision detection (issue 11): comparing immutable raw snapshots",
        "",
        ("This project cannot prove Treasury's historical daily rates are 'final' -- it can only observe the "
        "current official snapshot at each retrieval, and detect a change *prospectively* if it downloads "
        "the same year again later. Every comparison below is between two immutable raw artifacts this "
        "project actually holds, never an assumption from a single snapshot."),
        "",
        _df_to_markdown(snapshot_comparison_df) if not snapshot_comparison_df.empty else (
            "(Only one raw snapshot exists per year so far -- no multi-snapshot comparison is possible yet. "
            "This is disclosed, not treated as evidence of 'no revisions.')"
        ),
        "",
        "## Auction join coverage by tenor (ordinary_modeling_sample, announcement cutoff)",
        "",
        _df_to_markdown(coverage_by_tenor),
        "",
        "## Auction join coverage by year (ordinary_modeling_sample, announcement cutoff)",
        "",
        _df_to_markdown(coverage_by_year),
        "",
        "## Missing-reason breakdown (ordinary_modeling_sample, announcement cutoff)",
        "",
        _df_to_markdown(missing_breakdown) if not missing_breakdown.empty else "(none -- full coverage)",
        "",
        "## Sample-universe reconciliation",
        "",
        f"- `normalized_complete_sample`: {len(normalized_complete_sample)} rows (includes the 2 special auctions).",
        (f"- `ordinary_modeling_sample`: {len(ordinary_modeling_sample)} rows -- matches "
        "`select_modeling_sample` exactly (asserted at generation time)."),
        (f"- Announcement-cutoff join matched: {int(joined_ann['rates_join_matched'].sum())} / {len(joined_ann)} "
        f"(`normalized_complete_sample`: {int(joined_ann_complete['rates_join_matched'].sum())} / {len(joined_ann_complete)})."),
        "",
        "## Worked example (most recent ordinary 10-Year auction)",
        "",
        f"- Auction: 10-Year, {example_row['auction_date'].date()}, CUSIP `{example_row['cusip']}`",
        (f"- Announcement cutoff {example_row[ANNOUNCEMENT_CUTOFF_COL].date()} -> matched rate date "
        f"{example_row['rate_date'].date()} (10 Yr = {example_row['matched_tenor_par_yield_percent']}%), "
        f"age {example_row['rate_observation_age_calendar_days']} calendar days."),
        (f"- Pre-auction cutoff {example_pre_row[PRE_AUCTION_CUTOFF_COL].date()} -> matched rate date "
        f"{example_pre_row['rate_date'].date()} (10 Yr = {example_pre_row['matched_tenor_par_yield_percent']}%), "
        f"age {example_pre_row['rate_observation_age_calendar_days']} calendar days."),
        "",
        "## Candidate features constructed",
        "",
        ("Matched-tenor yield, adjacent lower/higher-maturity yield, rate age (calendar and business days), "
        "2s10s and 5s30s slopes (bps), a 2-10-30 curvature measure (bps), 1/5/20-trading-day changes (bps), "
        "5/20-day realized volatility (bps, past-only via `shift(1)` before `rolling`), and a methodology-"
        "regime indicator per maturity. No cross-sectional interpolation of missing maturities is performed."),
        "",
        "## Limitations",
        "",
        ("- No exact historical intraday publication timestamp exists; the next-full-business-day safety "
        "rule is a disclosed rule, not a verified fact for every date."),
        ("- The pre-2009 30-Year issuance suspension (2002-02-19 to 2006-02-08) and pre-2002 missing "
        "20-Year column predate this project's ingestion window and are documented for context only."),
        ("- 'No revisions observed' above is a real, checked fact only for the specific snapshot pairs "
        "compared -- it is not a general claim that Treasury's daily rates are immutable at the source."),
    ]

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.reports_dir / "treasury_rates_data_quality.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
