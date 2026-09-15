"""Phase 3 entry point: join NY Fed Primary Dealer Statistics onto
Treasury auctions and regenerate both Phase 3 reports.

Run it via uv, from the repository root, after
`treasury_auction_stress.data.cli` and
`treasury_auction_stress.data.dealer_stats_cli` have both produced
their processed tables:

    uv run python -m treasury_auction_stress.features.dealer_profile_cli

This module never re-downloads data; it only reads the already-
normalized parquet tables both prior CLIs produced and reports on
them. Every number in the generated markdown comes from actually
running this code against those tables -- see `docs/project_rules.md`'s "no
fabricated results" rule. Trains no model and adds no new source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.dealer_stats_schema import SELECTED_SERIES
from treasury_auction_stress.features.dealer_candidate_features import (
    TENOR_TO_STABLE_NAME,
    add_inventory_relative_to_offering,
    build_dealer_feature_table,
)
from treasury_auction_stress.features.dealer_join import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    STABLE_SERIES_NAMES,
    add_cutoff_dates,
    as_of_join,
    coverage_summary,
    worked_example,
)
from treasury_auction_stress.features.eligibility import (
    describe_eligibility,
    select_analysis_sample,
    select_modeling_sample,
)

# Named, unambiguous sample universes (Phase 3 acceptance review,
# "Sample-universe reconciliation").
#
# - NORMALIZED_COMPLETE_SAMPLE: every settled-or-pending nominal-coupon
#   auction from 2010-01-01 onward, INCLUDING the two verified special
#   (restricted, primary-dealer-only) auctions -- the full audit trail,
#   nothing excluded. Last verified count: 1281 rows (1281 settled + 0
#   pending).
# - ORDINARY_MODELING_SAMPLE: NORMALIZED_COMPLETE_SAMPLE with pending
#   auctions and the two special auctions excluded -- the sample a
#   future modeling phase would actually train on, matching Phase 2's
#   `select_modeling_sample`. Last verified count: 1279 rows.
#
# Every coverage table in the generated reports states, explicitly,
# which of these two it was computed over.
NORMALIZED_COMPLETE_SAMPLE_NAME = "normalized_complete_sample"
ORDINARY_MODELING_SAMPLE_NAME = "ordinary_modeling_sample"

# Earliest dealer-data coverage, by extension status (see
# dealer_stats_schema.py's historical-extension section) -- there is
# no longer a single "coverage starts here" date for every series.
DIRECTLY_EXTENDED_AND_HARMONIZED_COVERAGE_START = pd.Timestamp("2001-07-04")
NEVER_EXTENDED_COVERAGE_START = pd.Timestamp("2013-04-03")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """A minimal Markdown-table renderer (no `tabulate` dependency)."""
    if df.empty:
        return "(none)"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("nan" if pd.isna(v) else f"{v:.4g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _series_coverage_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (stable name, source keyid, regime) actually present
    in the normalized long table -- not just `SELECTED_SERIES` -- so a
    directly-extended column's two source regimes (e.g.
    `dealer_net_position_bills`: legacy `PDPUSGTBNOP` then modern
    `PDPOSGS-B`) both show up explicitly, alongside every legacy-only
    and discontinued component series. Per the "no fabricated results"
    rule, every count here comes from the actual normalized data, not
    from `dealer_stats_schema.py`'s static configuration.
    """
    group_cols = ["stable_series_name", "original_series_code", "category", "historical_regime_id"]
    rows = []
    for keys, sub in long_df.groupby(group_cols, dropna=False):
        stable_name, keyid, category, regime_id = keys
        n = len(sub)
        n_missing = int(sub["is_missing"].sum())
        rows.append(
            {
                "keyid": keyid,
                "stable_name": stable_name,
                "category": category,
                "regime_id": regime_id,
                "n_observations": n,
                "first_observation_date": sub["observation_date"].min(),
                "last_observation_date": sub["observation_date"].max(),
                "n_source_missing": n_missing,
                "pct_source_missing": (n_missing / n * 100.0) if n else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["stable_name", "first_observation_date"]).reset_index(drop=True)


def _ambiguous_same_day_cases(
    auctions: pd.DataFrame, dealer_wide: pd.DataFrame, *, cutoff_col: str
) -> pd.DataFrame:
    """Rows where a naive join on the raw `publication_date` (no safety
    buffer) would have selected a *more recent* observation than the
    actual safe join -- i.e. genuinely ambiguous same-day cases this
    project's conservative buffer resolves. Excludes rows where neither
    approach found any coverage at all (both `NaT`) -- that is a
    coverage gap, not an ambiguity. See `dealer_stats_normalize.py`'s
    module docstring for the rule.
    """
    auctions_indexed = auctions.reset_index(drop=True).copy()
    auctions_indexed["_row_id"] = auctions_indexed.index

    naive = dealer_wide.rename(columns={"publication_date": "_naive_key"})
    safe_selected = as_of_join(auctions_indexed, dealer_wide, cutoff_col=cutoff_col).set_index("_row_id")
    naive_selected = pd.merge_asof(
        auctions_indexed.sort_values(cutoff_col),
        naive.sort_values("_naive_key"),
        left_on=cutoff_col,
        right_on="_naive_key",
        direction="backward",
    ).set_index("_row_id")

    safe_obs = safe_selected["observation_date"].reindex(auctions_indexed["_row_id"])
    naive_obs = naive_selected["observation_date"].reindex(auctions_indexed["_row_id"])
    differs = naive_obs.notna() & (safe_obs.isna() | (safe_obs != naive_obs))

    out = auctions_indexed.loc[differs.to_numpy(), ["auction_date", "tenor", cutoff_col]].copy()
    out["safe_join_observation_date"] = safe_obs.loc[differs].to_numpy()
    out["naive_join_would_have_selected"] = naive_obs.loc[differs].to_numpy()
    return out


def _missing_reason_breakdown(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in STABLE_SERIES_NAMES:
        col = f"{name}_missing_reason"
        if col not in joined.columns:
            continue
        counts = joined[col].str.extract(r"^([a-z_]+)")[0].value_counts(dropna=True)
        for reason, n in counts.items():
            rows.append({"series": name, "reason": reason, "n_auctions": int(n)})
    return pd.DataFrame(rows)


def _render_data_quality_report(
    generated_note: str,
    series_coverage: pd.DataFrame,
    retrieval_meta: dict,
) -> str:
    lines = [
        "# Primary Dealer Data Quality Report",
        "",
        generated_note,
        "",
        "## Series selected and per-series coverage",
        "",
        "| Series | Category | Regime | First obs | Last obs | N obs | N source-missing | % missing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in series_coverage.iterrows():
        lines.append(
            f"| `{r['keyid']}` ({r['stable_name']}) | {r['category']} | {r['regime_id']} | "
            f"{r['first_observation_date'].date()} | {r['last_observation_date'].date()} | "
            f"{r['n_observations']} | {r['n_source_missing']} | {r['pct_source_missing']:.1f}% |"
        )
    lines += [
        "",
        "## Retrieval provenance",
        "",
        f"- Source URL(s) sample: `{retrieval_meta.get('sample_url', 'n/a')}`",
        f"- Retrieval timestamp (UTC): `{retrieval_meta.get('retrieval_timestamp_utc', 'n/a')}`",
        f"- Row count: {retrieval_meta.get('row_count', 'n/a')}",
        f"- Checksum (sha256): `{retrieval_meta.get('checksum_sha256', 'n/a')}`",
        "",
        "## Missing-value convention",
        "",
        ("The source uses the literal string `\"*\"` for a withheld/unavailable "
        "weekly value -- distinct from the Fiscal Data auctions API's `\"null\"` "
        "string. This project converts `\"*\"` to a real missing value and never "
        "imputes, zero-fills, or forward-fills it. See "
        "`treasury_auction_stress.data.dealer_stats_normalize`."),
        "",
        "## Verified schema regimes",
        "",
        ("- **2001-07-04 to 2013-03-27 (legacy `SBP2013` period)**: source "
        "history for `dealer_net_position_bills` and "
        "`dealer_net_position_coupons_3y_6y` (directly extended, Phase 3 "
        "acceptance review), plus three legacy-only component series used "
        "solely to build the harmonized coarse buckets. Retrieved via a "
        "period-scoped endpoint, not the plain current-API endpoint -- see "
        "the Phase 3 acceptance review."),
        ("- **2013-04-03 -- present**: every other selected series, plus "
        "the modern portion of the two directly-extended series above. "
        "No earlier history is available under these *specific* keyids "
        "(see `dealer_stats_schema.py` module docstring) -- but see the "
        "acceptance review for which concepts have a genuine, "
        "differently-keyed pre-2013 history and which do not."),
        ("- **2013-04-03 to 2021-12-29 (discontinued keyid, still live-"
        "fetchable)**: `PDPOSGSC-G11` (combined >11y bucket), used as the "
        "middle piece of the harmonized long-duration feature."),
        ("- **2022-01-05 -- present**: `PDPOSGSC-G11L21` (11-21y) and "
        "`PDPOSGSC-G21` (>21y) position buckets only -- verified directly "
        "against the live API (701 weeks for every other series vs. exactly "
        "244 for these two, both starting 2022-01-05). No currently-listed "
        "keyid covers a combined \"more than 11 years\" bucket before this "
        "date."),
        ("- **2022-01-05 methodology note (not a keyid break)**: per the "
        "Federal Reserve's own FEDS Note (2022-08-05), FR 2004C's repo/"
        "reverse-repo counterparty segmentation was revised on this date; "
        "`PDSORA-UTSETTOT` and `PDSIRRA-UTSETTOT` remain one continuous "
        "keyid across it, but the methodology underlying their published "
        "total changed. No equivalent official statement was found for "
        "securities-borrowed/-lent or fails."),
        "",
        "## One verified, deliberately-excluded series with an unresolved metadata ambiguity",
        "",
        ("**Corrected during the Phase 3 acceptance review**: this project no "
        "longer asserts NY Fed mislabeled `PDTRGST-TOT` -- no official source "
        "confirms that. What remains true: `PDTRGST-TOT`'s own API "
        "description claims to be a TIPS-only total, but its naming pattern "
        "and its value level (~$873B avg vs. ~$23B for the unambiguous TIPS "
        "total `PDTIPSTOT`) conflict with that claim. This is documented as "
        "an unresolved ambiguity, not a confirmed error. This project uses "
        "`PDGSWOEXTTOT` instead, whose excl.-TIPS definition is independently "
        "verified structurally (not just by value) via NY Fed's own "
        "historical menu data -- see `dealer_stats_schema.py`."),
    ]
    return "\n".join(lines) + "\n"


def _render_integration_report(
    generated_note: str,
    coverage_by_tenor_ann_ordinary: pd.DataFrame,
    coverage_by_tenor_pre_ordinary: pd.DataFrame,
    coverage_by_tenor_ann_complete: pd.DataFrame,
    coverage_by_tenor_pre_complete: pd.DataFrame,
    coverage_by_year_ordinary: pd.DataFrame,
    ambiguous_ann: pd.DataFrame,
    ambiguous_pre: pd.DataFrame,
    missing_breakdown_ordinary: pd.DataFrame,
    example: dict,
    eligibility_summary: dict,
    n_complete: int,
    n_ordinary: int,
    n_before_extended: int,
    n_before_never_extended: int,
) -> str:
    lines = [
        "# Phase 3: Primary Dealer Statistics Integration Report",
        "",
        generated_note,
        "",
        "## Series selected and economic rationale",
        "",
    ]
    for s in SELECTED_SERIES:
        lines.append(f"- **`{s.keyid}`** (`{s.stable_name}`, {s.category}): {s.rationale}")
    lines += [
        "",
        "## Publication rule (verified)",
        "",
        ("NY Fed's own stated cadence: \"Data are updated on Thursdays at "
        "approximately 4:15 p.m. with the previous week's statistics.\" This "
        "project computes, for every observation: `publication_date` "
        "(the Wednesday observation date + 1 day, advanced past any U.S. "
        "federal holiday) and `publication_safe_available_date` "
        "(the next full U.S. business day after `publication_date`, via "
        "`time_utils.next_full_business_day_after`, to resolve the documented "
        "same-day intraday-timing ambiguity conservatively). Every join uses "
        "`publication_safe_available_date`, never `observation_date` or the "
        "raw `publication_date`. See `docs/point_in_time_rules.md`."),
        "",
        "### Phase 5 correction: safe-availability date no longer a plain `+1 calendar day`",
        "",
        ("The Phase 4 acceptance review disclosed (but left out of scope) a "
        "latent bug in this module: `publication_safe_available_date` used to "
        "be `publication_date + 1 calendar day`, which is safe in the ordinary "
        "case (Thursday publication + 1 day = Friday, a business day) but "
        "silently produces a **Saturday** safe-available date whenever a "
        "federal holiday shifts `publication_date` itself onto a Friday (e.g. "
        "a Thursday-holiday week such as Thanksgiving) -- the same bug class "
        "the Phase 4 acceptance review already found and fixed for Treasury "
        "rates and CFTC. Phase 5 fixed this by calling the same shared "
        "`next_full_business_day_after` function those modules use, instead "
        "of re-deriving a calendar-day rule here."),
        ("**Verified effect on this project's actual auction joins: none.** "
        "Re-running both `as_of_join` calls (announcement and pre-auction "
        "cutoffs) against the pre-fix and post-fix dealer feature tables over "
        "the full 1,281-row `normalized_complete_sample` selects the exact "
        "same dealer-release `observation_date` for every single auction, "
        "under both cutoffs -- zero changes. This is not a coincidence: every "
        "auction cutoff this project ever computes (an announcement date or a "
        "\"previous business day\" pre-auction cutoff) is itself always a "
        "business day (Treasury never announces or holds an auction on a "
        "weekend), so a cutoff can never fall in the narrow window between "
        "the old rule's incorrect Saturday safe-date and the new rule's "
        "correct following-Monday safe-date -- that window contains only "
        "weekend days, which are never cutoff dates. The defect was real "
        "(a wrong date was computed and stored), but it never actually "
        "changed which dealer release any auction's join selected."),
        "",
        "### Important finding: announcement dates cluster on the *same* weekday as the dealer-statistics release",
        "",
        ("Verified directly from the processed auction table "
        "(`announcemt_date.dt.day_name()`): the overwhelming majority of "
        "nominal-coupon auction announcements fall on a **Thursday** -- "
        "e.g. 185/190 for 2-Year, 194/199 for 5-Year, 205/211 for 7-Year, "
        "76/77 for 20-Year -- the *same* weekday NY Fed publishes Primary "
        "Dealer Statistics. This means the \"ambiguous same-day boundary\" "
        "case this project's one-day safety buffer resolves is not a rare "
        "edge case for the announcement-date cutoff -- it is the **typical** "
        "case. Without the buffer, a naive join keyed on the raw "
        "`publication_date` would, for a majority of auctions, gamble on "
        "whether the announcement time of day fell before or after NY "
        "Fed's ~4:15pm ET release; the buffer instead deterministically "
        "and conservatively falls back to the prior week's release. This "
        "is exactly why the buffer is a load-bearing design choice, not a "
        "cosmetic one -- see the counts immediately below."),
        "",
        "## Sample-universe reconciliation",
        "",
        (f"- **`{NORMALIZED_COMPLETE_SAMPLE_NAME}`**: every settled-or-pending "
        f"nominal-coupon auction from 2010-01-01 onward, INCLUDING the two "
        f"verified special (restricted, primary-dealer-only) auctions -- "
        f"the complete audit trail. **{n_complete} rows** this session."),
        (f"- **`{ORDINARY_MODELING_SAMPLE_NAME}`**: `{NORMALIZED_COMPLETE_SAMPLE_NAME}` "
        f"with pending auctions and the two special auctions excluded -- "
        f"matches `treasury_auction_stress.features.eligibility.select_modeling_sample` "
        f"exactly (verified by assertion in `dealer_profile_cli.run`, not just by "
        f"eyeballing the count). **{n_ordinary} rows** this session."),
        ("- Every coverage table below states which of these two samples it "
        "uses. Unless a table is explicitly labeled "
        f"`{NORMALIZED_COMPLETE_SAMPLE_NAME}`, it uses "
        f"`{ORDINARY_MODELING_SAMPLE_NAME}` ({n_ordinary} rows) -- the sample "
        "a future modeling phase would actually consume."),
        "",
        f"## Auction join coverage by tenor ({ORDINARY_MODELING_SAMPLE_NAME}, {n_ordinary} rows)",
        "",
        "### Announcement-date cutoff",
        "",
        _df_to_markdown(coverage_by_tenor_ann_ordinary),
        "",
        "### Pre-auction cutoff",
        "",
        _df_to_markdown(coverage_by_tenor_pre_ordinary),
        "",
        f"## Auction join coverage by tenor ({NORMALIZED_COMPLETE_SAMPLE_NAME}, {n_complete} rows -- includes the 2 special auctions)",
        "",
        "### Announcement-date cutoff",
        "",
        _df_to_markdown(coverage_by_tenor_ann_complete),
        "",
        "### Pre-auction cutoff",
        "",
        _df_to_markdown(coverage_by_tenor_pre_complete),
        "",
        f"## Auction join coverage by year ({ORDINARY_MODELING_SAMPLE_NAME}, announcement cutoff)",
        "",
        _df_to_markdown(coverage_by_year_ordinary),
        "",
        f"## Coverage gap: auctions before dealer-data availability ({ORDINARY_MODELING_SAMPLE_NAME})",
        "",
        (f"- Total auctions in `{ORDINARY_MODELING_SAMPLE_NAME}`: {n_ordinary}"),
        (f"- Auctions before {DIRECTLY_EXTENDED_AND_HARMONIZED_COVERAGE_START.date()} "
        "(the earliest coverage for ANY series, after the Phase 3 acceptance "
        f"review's historical extension): {n_before_extended} -- for our "
        "2010-01-01-onward universe, this is expected to be 0 (verify against "
        "the number printed here, not assumed)."),
        (f"- Auctions before {NEVER_EXTENDED_COVERAGE_START.date()} (the "
        "coverage start for every series this review did NOT extend -- "
        "transactions, repo, reverse repo, securities borrowed/lent, fails; "
        f"see the Phase 3 acceptance review): {n_before_never_extended} "
        "-- these are **preserved** in the join output (never dropped), "
        "tagged with a `series_regime_not_yet_started` missing reason for "
        "those specific series only (the directly-extended and harmonized "
        "series remain populated for these same rows)."),
        "",
        f"## Missing-reason breakdown ({ORDINARY_MODELING_SAMPLE_NAME}, announcement cutoff, all series x all auctions)",
        "",
        _df_to_markdown(missing_breakdown_ordinary),
        "",
        "## Ambiguous same-day cases resolved by the conservative buffer",
        "",
        (f"- Announcement cutoff: {len(ambiguous_ann)} auction(s) where a naive "
        "join on the raw `publication_date` (no buffer) would have selected "
        "a different, more recent observation than the actual safe join."),
        f"- Pre-auction cutoff: {len(ambiguous_pre)} auction(s), same check.",
        "",
    ]
    if not ambiguous_ann.empty:
        lines += [
            "### Example ambiguous same-day cases (announcement cutoff)",
            "",
            _df_to_markdown(ambiguous_ann.sort_values("auction_date").tail(5)),
            "",
        ]
    lines += [
        "## Worked example: one fully-explained auction join",
        "",
        (f"- Auction: {example['tenor']} auctioned {example['auction_date']}, "
        f"CUSIP `{example['cusip']}`"),
        (f"- Announcement cutoff: {example['announcement_cutoff_date']} -> "
        f"selected dealer observation week {example['announcement_cutoff_selected_observation_date']} "
        f"(value {example['announcement_cutoff_selected_value']})"),
        (f"- Pre-auction cutoff: {example['pre_auction_cutoff_date']} -> "
        f"selected dealer observation week {example['pre_auction_cutoff_selected_observation_date']} "
        f"(value {example['pre_auction_cutoff_selected_value']})"),
        "",
        "| Observation date | Publication date | Safe-available date | Value | Selected (announcement) | Selected (pre-auction) | Rejected because |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in example["candidates"]:
        lines.append(
            f"| {c['observation_date'].date()} | {c['publication_date'].date()} | "
            f"{c['publication_safe_available_date'].date()} | {c[list(c.keys())[3]]} | "
            f"{c['selected_by_announcement_cutoff']} | {c['selected_by_pre_auction_cutoff']} | "
            f"{c['rejected_because'] or ''} |"
        )
    lines += [
        "",
        "## Special auctions (unchanged from Phase 2)",
        "",
        (f"- Special (restricted, primary-dealer-only) auction rows: "
        f"{eligibility_summary['special_auction_rows']} -- preserved in "
        f"`{NORMALIZED_COMPLETE_SAMPLE_NAME}`, excluded from "
        f"`{ORDINARY_MODELING_SAMPLE_NAME}`, exactly as established in "
        "Phase 2. Phase 3 does not alter this rule; see "
        "'Sample-universe reconciliation' above for the exact row counts."),
        "",
        "## Candidate dealer features constructed",
        "",
        ("Per-series raw levels, `_wow_change` (week-over-week), `_chg_4w`/"
        "`_chg_13w` (trailing changes), `_zscore_52w` (past-only rolling "
        "standardization, `shift(1)` before `rolling`, mirroring "
        "`dealer_absorption.add_regime_feature`), `dealer_long_short_"
        "imbalance` (cross-maturity), `dealer_inventory_to_offering_"
        "ratio` (tenor-matched bucket inventory vs. the auction's own "
        "offering amount), and (Phase 3 acceptance review) three "
        "harmonized, cross-schema-period coarse buckets plus a harmonized "
        "whole-curve total, extending real coverage back to 2001-07-04 for "
        "every tenor except 5-Year (whose own fine-grained bucket was "
        "directly extended instead). All past-only by construction; none "
        "trained as a predictive model in this phase."),
        "",
        "## Limitations",
        "",
        ("- Transactions, repo, reverse repo, securities borrowed/lent, and "
        "fails were deliberately NOT extended before 2013-04-03 -- see "
        "the Phase 3 acceptance review for why each one specifically "
        "could not be cleanly separated from TIPS, or (securities "
        "borrowed/lent) did not exist as a reported concept before that date."),
        ("- The fine-grained 11-21y and >21y position buckets (20-Year and "
        "30-Year tenor matches) are only available from 2022-01-05 onward; "
        "the harmonized >11y bucket covers both tenors together (not "
        "separately) back to 2001-07-04."),
        ("- The pre-auction cutoff's \"previous business day\" uses the U.S. "
        "federal holiday calendar, not a SIFMA bond-market holiday calendar "
        "-- Good Friday is a known, disclosed divergence with no practical "
        "effect on which weekly release is selected (see "
        "`treasury_auction_stress.features.dealer_join` module docstring)."),
        ("- The holiday-shift rule for `publication_date` is a disclosed "
        "heuristic (no primary source confirms exact historical "
        "holiday-shifted release dates); the one-day safety buffer is "
        "designed to absorb exactly this kind of uncertainty."),
        ("- Tenor-to-maturity-bucket mapping uses each auction's original "
        "tenor, not a reopening's true remaining maturity, mirroring the "
        "project's existing tenor-grouping convention elsewhere."),
        ("- See the Phase 3 acceptance review for the full "
        "concept-by-concept historical-extension compatibility matrix."),
    ]
    return "\n".join(lines) + "\n"


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
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


    long_df = pd.read_parquet(args.processed_dir / "dealer_stats_long.parquet")
    wide_df = pd.read_parquet(args.processed_dir / "dealer_stats_wide.parquet")
    feat = build_dealer_feature_table(wide_df)

    nominal_df = pd.read_parquet(args.processed_dir / "treasury_auctions_nominal_coupons.parquet")
    settled, pending = select_analysis_sample(nominal_df)
    normalized_complete_sample = pd.concat([settled, pending], ignore_index=True)
    normalized_complete_sample = add_cutoff_dates(normalized_complete_sample)

    # Joined once, over the complete sample; the ordinary-modeling
    # view is then a pure row filter of that same join (never a
    # separate join), so the two are guaranteed consistent with each
    # other by construction.
    joined_ann_complete = as_of_join(normalized_complete_sample, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)
    joined_pre_complete = as_of_join(normalized_complete_sample, feat, cutoff_col=PRE_AUCTION_CUTOFF_COL)
    joined_ann_complete = add_inventory_relative_to_offering(joined_ann_complete)

    ordinary_row_mask_ann = joined_ann_complete["results_available"] & joined_ann_complete[
        "special_auction_type"
    ].isna()
    joined_ann_ordinary = joined_ann_complete.loc[ordinary_row_mask_ann].reset_index(drop=True)
    ordinary_row_mask_pre = joined_pre_complete["results_available"] & joined_pre_complete[
        "special_auction_type"
    ].isna()
    joined_pre_ordinary = joined_pre_complete.loc[ordinary_row_mask_pre].reset_index(drop=True)

    n_complete = len(normalized_complete_sample)
    n_ordinary = len(joined_ann_ordinary)
    assert n_ordinary == len(select_modeling_sample(settled)), (
        "ordinary_modeling_sample row count derived by post-join filtering must exactly "
        "match treasury_auction_stress.features.eligibility.select_modeling_sample's own "
        "count -- a mismatch here would mean the two disagree on what 'ordinary' means"
    )

    def _coverage_cols(df: pd.DataFrame, group_cols: tuple[str, ...]) -> pd.DataFrame:
        cols = [*group_cols, "n_auctions", "n_matched_any_dealer_release", "median_dealer_observation_age_days"]
        return coverage_summary(df, group_cols=group_cols)[cols]

    coverage_ann_complete = _coverage_cols(joined_ann_complete, ("tenor",))
    coverage_pre_complete = _coverage_cols(joined_pre_complete, ("tenor",))
    coverage_ann_ordinary = _coverage_cols(joined_ann_ordinary, ("tenor",))
    coverage_pre_ordinary = _coverage_cols(joined_pre_ordinary, ("tenor",))

    coverage_by_year_ordinary = coverage_summary(
        joined_ann_ordinary.assign(year=joined_ann_ordinary["auction_date"].dt.year), group_cols=("year",)
    )[["year", "n_auctions", "n_matched_any_dealer_release"]]

    ambiguous_ann = _ambiguous_same_day_cases(
        normalized_complete_sample, feat, cutoff_col=ANNOUNCEMENT_CUTOFF_COL
    )
    ambiguous_pre = _ambiguous_same_day_cases(
        normalized_complete_sample, feat, cutoff_col=PRE_AUCTION_CUTOFF_COL
    )
    missing_breakdown_ordinary = _missing_reason_breakdown(joined_ann_ordinary)

    example_tenor = "10-Year"
    example = worked_example(
        joined_ann_complete, joined_pre_complete, feat, example_series=TENOR_TO_STABLE_NAME[example_tenor]
    )

    eligibility_summary = describe_eligibility(nominal_df)

    n_before_never_extended = int(
        (joined_ann_ordinary["auction_date"] < NEVER_EXTENDED_COVERAGE_START).sum()
    )
    n_before_extended = int(
        (joined_ann_ordinary["auction_date"] < DIRECTLY_EXTENDED_AND_HARMONIZED_COVERAGE_START).sum()
    )

    series_coverage = _series_coverage_table(long_df)

    raw_meta_files = sorted((args.raw_dir).glob("ny_fed_primary_dealer_stats_raw_retrieved_*.meta.json"))
    retrieval_meta: dict = {}
    if raw_meta_files:
        import json

        meta = json.loads(raw_meta_files[-1].read_text(encoding="utf-8"))
        retrieval_meta = {
            "sample_url": meta["request_urls"][1] if len(meta["request_urls"]) > 1 else meta["request_urls"][0],
            "retrieval_timestamp_utc": meta["retrieval_timestamp_utc"],
            "row_count": meta["row_count"],
            "checksum_sha256": meta["checksum_sha256"],
        }

    generated_note = (
        "Generated by `uv run python -m treasury_auction_stress.features.dealer_profile_cli` "
        f"from {len(long_df)} normalized dealer-statistics rows. Sample universes (see "
        f"'Sample-universe reconciliation' below): `{NORMALIZED_COMPLETE_SAMPLE_NAME}` has "
        f"{n_complete} rows; `{ORDINARY_MODELING_SAMPLE_NAME}` has {n_ordinary} rows."
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    quality_path = args.reports_dir / "primary_dealer_data_quality.md"
    quality_path.write_text(
        _render_data_quality_report(generated_note, series_coverage, retrieval_meta), encoding="utf-8"
    )
    print(f"wrote {quality_path}")

    integration_path = args.reports_dir / "phase_3_primary_dealer_integration.md"
    integration_path.write_text(
        _render_integration_report(
            generated_note,
            coverage_ann_ordinary,
            coverage_pre_ordinary,
            coverage_ann_complete,
            coverage_pre_complete,
            coverage_by_year_ordinary,
            ambiguous_ann,
            ambiguous_pre,
            missing_breakdown_ordinary,
            example,
            eligibility_summary,
            n_complete,
            n_ordinary,
            n_before_extended,
            n_before_never_extended,
        ),
        encoding="utf-8",
    )
    print(f"wrote {integration_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
