"""Phase 4B entry point: join CFTC TFF Futures Only positioning onto
Treasury auctions and regenerate `artifacts/cftc_positioning_data_quality.md`.

Run it via uv, from the repository root, after
`treasury_auction_stress.data.cftc_cli` and
`treasury_auction_stress.data.cli` have both produced their processed
tables:

    uv run python -m treasury_auction_stress.features.cftc_profile_cli

Every number in the generated markdown comes from actually running
this code -- see `docs/project_rules.md`'s "no fabricated results" rule. Trains no
model, adds no new source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.cftc_release_calendar import CFTC_REPORT_OVERRIDES
from treasury_auction_stress.data.cftc_schema import (
    CONTRACT_REGISTRY,
    TENOR_PRIMARY_CONTRACT_CODE,
    TENORS_WITH_NO_CONTRACT,
)
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.cftc_join import (
    DIRECT_CONTRACT_STATUS_AVAILABLE,
    DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE,
    DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR,
    add_tenor_matched_positioning_features,
    as_of_join,
)
from treasury_auction_stress.features.eligibility import (
    select_analysis_sample,
    select_modeling_sample,
)

RECONCILIATION_NO_REPORT = "no_safely_available_cftc_report_before_cutoff"
RECONCILIATION_NO_DIRECT_CONTRACT = "report_available_no_direct_contract_for_tenor"
RECONCILIATION_DIRECT_AVAILABLE = "report_available_direct_contract_available"
RECONCILIATION_DIRECT_NO_DATA = "report_available_direct_contract_mapped_but_data_unavailable"


def row_level_reconciliation(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Every auction row falls into exactly one of 4 mutually exclusive,
    exhaustive states (Phase 4 acceptance review, issue 8). The counts
    must sum exactly to `len(joined_df)`.
    """
    state = pd.Series(RECONCILIATION_NO_REPORT, index=joined_df.index, dtype=object)
    available = joined_df["cftc_report_available"]
    state.loc[available & (joined_df["direct_contract_mapping_status"] == DIRECT_CONTRACT_STATUS_AVAILABLE)] = RECONCILIATION_DIRECT_AVAILABLE
    state.loc[available & (joined_df["direct_contract_mapping_status"] == DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR)] = RECONCILIATION_NO_DIRECT_CONTRACT
    state.loc[available & (joined_df["direct_contract_mapping_status"] == DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE)] = RECONCILIATION_DIRECT_NO_DATA
    counts = state.value_counts()
    assert counts.sum() == len(joined_df), f"reconciliation states do not sum to {len(joined_df)}: {counts.to_dict()}"
    return pd.DataFrame({"state": counts.index, "n_auctions": counts.to_numpy()})


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
    args = parser.parse_args(argv)

    long_df = pd.read_parquet(args.processed_dir / "cftc_positioning_long.parquet")
    wide_df = pd.read_parquet(args.processed_dir / "cftc_positioning_wide.parquet")

    nominal_df = pd.read_parquet(args.processed_dir / "treasury_auctions_nominal_coupons.parquet")
    settled, pending = select_analysis_sample(nominal_df)
    normalized_complete_sample = add_cutoff_dates(pd.concat([settled, pending], ignore_index=True))
    ordinary_modeling_sample = add_cutoff_dates(select_modeling_sample(settled))
    assert len(normalized_complete_sample) == len(settled) + len(pending)
    assert len(ordinary_modeling_sample) == len(select_modeling_sample(settled))

    joined_ann = as_of_join(ordinary_modeling_sample, wide_df, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    joined_pre = as_of_join(ordinary_modeling_sample, wide_df, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    joined_ann = add_tenor_matched_positioning_features(joined_ann)
    joined_pre = add_tenor_matched_positioning_features(joined_pre)

    joined_ann_complete = as_of_join(normalized_complete_sample, wide_df, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    joined_ann_complete = add_tenor_matched_positioning_features(joined_ann_complete)

    coverage_by_tenor = (
        joined_ann.groupby("tenor")
        .agg(
            n_auctions=("cusip", "size"),
            n_report_available=("cftc_report_available", "sum"),
            n_direct_contract_available=("direct_contract_mapping_status", lambda s: (s == DIRECT_CONTRACT_STATUS_AVAILABLE).sum()),
        )
        .reset_index()
    )
    coverage_by_year = (
        joined_ann.assign(year=joined_ann["auction_date"].dt.year)
        .groupby("year")
        .agg(n_auctions=("cusip", "size"), n_report_available=("cftc_report_available", "sum"))
        .reset_index()
    )

    reconciliation = row_level_reconciliation(joined_ann)

    example_row = joined_ann.loc[
        joined_ann["matched_contract_code"].notna()
    ].sort_values("auction_date").iloc[-1]
    example_pre_row = joined_pre.loc[
        (joined_pre["cusip"] == example_row["cusip"]) & (joined_pre["auction_date"] == example_row["auction_date"])
    ].iloc[0]

    contract_registry_rows = pd.DataFrame(
        [
            {
                "code": c.code,
                "official_name": c.official_name,
                "first_verified_observation": c.first_verified_observation,
                "tenor_role": c.tenor_role,
            }
            for c in CONTRACT_REGISTRY
        ]
    )

    override_rows = pd.DataFrame(
        [
            {
                "observation_date": o.observation_date,
                "actual_publication_date": o.actual_publication_date,
                "exception_type": o.exception_type,
                "precision": o.precision,
            }
            for o in CFTC_REPORT_OVERRIDES
        ]
    )

    lines = [
        "# CFTC Traders in Financial Futures (TFF), Futures Only -- Data Quality Report",
        "",
        (f"Generated by `uv run python -m treasury_auction_stress.features.cftc_profile_cli` "
        f"from {len(long_df)} normalized positioning rows ({long_df['report_date'].min().date()} to "
        f"{long_df['report_date'].max().date()}, full retained source history -- see the Phase 4 "
        f"acceptance review, issue 8, for why this is no longer truncated to 2010-01-01) and "
        f"{len(ordinary_modeling_sample)} ordinary_modeling_sample auctions "
        f"({len(normalized_complete_sample)} in normalized_complete_sample). "
        "**Status: implemented and accepted** -- see the Phase 4 acceptance review."),
        "",
        "## Source and dataset (verified live 2026-09-11)",
        "",
        "- Socrata Public Reporting Environment: `https://publicreporting.cftc.gov`, dataset id `gpe5-46if` (**Futures Only**, verified via `futonly_or_combined == \"FutOnly\"` on every row).",
        "- The Futures-and-Options **Combined** dataset (`yw9f-hn96`) and the Legacy/Disaggregated COT datasets exist but are never queried by this project.",
        "- NOT obtained via FRED/ALFRED -- see `docs/data_source_governance.md`.",
        "- CFTC's `Dealer/Intermediary` category is a distinct, broader regulatory classification from NY Fed's designated primary dealers (Phase 3) -- never conflated.",
        "",
        "## Verified contract registry",
        "",
        _df_to_markdown(contract_registry_rows),
        "",
        (f"3-Year and 7-Year have **no** Treasury-note futures contract at all -- verified by complete "
        f"absence from a live query, not assumed (`TENORS_WITH_NO_CONTRACT = {TENORS_WITH_NO_CONTRACT}`). "
        f"20-Year has no *direct* contract either; only bond-future context "
        f"(`TENOR_PRIMARY_CONTRACT_CODE` omits it: {sorted(TENOR_PRIMARY_CONTRACT_CODE)}). "
        "**This is a property of the tenor, not a source-coverage failure** -- see the separated "
        "`cftc_report_available` / `direct_contract_mapping_status` fields below (acceptance review, issue 9)."),
        "",
        "## Release-calendar disruption per-report overrides (acceptance review, issue 10)",
        "",
        (f"{len(override_rows)} individual reports across 3 disruption periods now have their own "
        "actual (or, where officially conflicting, conservatively-resolved) publication date -- never a "
        "single bulk end-of-window date. See `cftc_release_calendar.py`'s module docstring for the exact "
        "source and reconstruction method behind each precision label."),
        "",
        _df_to_markdown(override_rows),
        "",
        ("Ordinary weeks use a documented standard rule (Tuesday observation -> Friday publication, "
        "3:30pm ET), with safe availability computed as the **next full U.S. business day** after "
        "publication (never a fixed `+1 calendar day`, which could land on a Saturday) -- flagged "
        "`documented_standard_rule_inferred_for_this_week`. A corrected finding: the original "
        "implementation's 2023 ION-incident window stopped 3 weeks too early (2023-02-21 instead of "
        "2023-03-14), which would have joined those 3 weeks' auctions to a report that had not yet "
        "actually published -- fixed by extending the override table to all 7 affected reports."),
        "",
        "## Auction join coverage by tenor (ordinary_modeling_sample, announcement cutoff)",
        "",
        _df_to_markdown(coverage_by_tenor),
        "",
        "## Auction join coverage by year (ordinary_modeling_sample, announcement cutoff)",
        "",
        _df_to_markdown(coverage_by_year),
        "",
        "## Row-level coverage reconciliation (ordinary_modeling_sample, announcement cutoff)",
        "",
        (f"Every one of the {len(joined_ann)} auctions falls into exactly one of 4 mutually exclusive, "
        f"exhaustive states -- the counts below sum exactly to {len(joined_ann)} (asserted at generation time)."),
        "",
        _df_to_markdown(reconciliation),
        "",
        "## Sample-universe reconciliation",
        "",
        f"- `normalized_complete_sample`: {len(normalized_complete_sample)} rows (includes the 2 special auctions).",
        (f"- `ordinary_modeling_sample`: {len(ordinary_modeling_sample)} rows -- matches "
        "`select_modeling_sample` exactly (asserted at generation time)."),
        (f"- **Any-report source coverage** (`cftc_report_available`, independent of tenor): "
        f"{int(joined_ann['cftc_report_available'].sum())} / {len(joined_ann)} "
        f"(`normalized_complete_sample`: {int(joined_ann_complete['cftc_report_available'].sum())} / {len(joined_ann_complete)})."),
        (f"- **Direct-contract mapping available** (a strictly narrower, tenor-dependent property -- "
        f"never conflated with the line above): "
        f"{int((joined_ann['direct_contract_mapping_status'] == DIRECT_CONTRACT_STATUS_AVAILABLE).sum())} / {len(joined_ann)} "
        f"(3-Year/7-Year/20-Year auctions structurally never get one -- see above)."),
        "",
        "## Worked example (most recent ordinary auction with a direct-match contract)",
        "",
        f"- Auction: {example_row['tenor']}, {example_row['auction_date'].date()}, CUSIP `{example_row['cusip']}`",
        (f"- Matched contract: `{example_row['matched_contract_code']}` "
        f"({example_row['matched_contract_official_name']}); secondary context: "
        f"{example_row['secondary_context_contract_codes'] or '(none)'}"),
        (f"- Announcement cutoff {example_row[ANNOUNCEMENT_CUTOFF_COL].date()} -> matched report week "
        f"{example_row['report_date'].date()} (dealer net = {example_row['matched_dealer_net_contracts']} contracts), "
        f"age {example_row['cftc_observation_age_calendar_days']} calendar days."),
        (f"- Pre-auction cutoff {example_pre_row[PRE_AUCTION_CUTOFF_COL].date()} -> matched report week "
        f"{example_pre_row['report_date'].date()} (dealer net = {example_pre_row['matched_dealer_net_contracts']} contracts), "
        f"age {example_pre_row['cftc_observation_age_calendar_days']} calendar days."),
        "",
        "## Candidate features constructed",
        "",
        ("Per-category net directional positions (long minus short, contracts -- spreading always kept "
        "separate), per-category share of open interest, 1- and 4-observation-week changes in net position "
        "(past-only, computed within each contract's own series), CFTC-reported concentration ratios passed "
        "through unmodified, and (for each auction's own tenor) the primary direct-match contract's features "
        "selected explicitly rather than averaged across contracts. No dollar/notional/DV01 relabeling; no "
        "full-sample z-score or percentile anywhere in this module."),
        "",
        "## Limitations",
        "",
        ("- No complete historical list of exact CFTC release timestamps exists for *ordinary* weeks; the "
        "3:30pm/next-full-business-day rule is a disclosed convention, not a verified fact for every "
        "individual week."),
        ("- The 2018-2019 shutdown's intermediate catch-up dates are reconstructed (not individually "
        "re-verified) from a documented cadence rule -- self-consistent with the independently confirmed "
        "start and end dates, but not a literal published table."),
        ("- For 7 of the 2025 shutdown's affected reports, two official CFTC press releases give "
        "conflicting publication dates; this project uses the later (more conservative) of the two without "
        "independent confirmation of which was actually followed."),
        "- 3-Year, 7-Year, and 20-Year auctions have no direct futures-contract positioning feature by construction, not by an ingestion gap -- this is never counted as a source-coverage failure (see the row-level reconciliation above).",
    ]

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.reports_dir / "cftc_positioning_data_quality.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
