"""Compute and render the Phase 1 data-quality report.

Every number in the rendered report comes from actually running this
code against actual data -- nothing here is a hand-typed example
value. If a number looks wrong, the fix belongs in the pipeline code
or the underlying data, not in this report.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from treasury_auction_stress.data.normalize import (
    CANDIDATE_PRIMARY_KEY,
    find_duplicate_keys,
    nominal_coupon_subset,
)
from treasury_auction_stress.data.schema import NOMINAL_COUPON_TENORS

CANDIDATE_TARGET_FIELDS: tuple[str, ...] = (
    "offering_amt",
    "total_accepted",
    "total_tendered",
    "bid_to_cover_ratio",
    "primary_dealer_accepted",
    "direct_bidder_accepted",
    "indirect_bidder_accepted",
    "noncomp_accepted",
)

# A gap between consecutive auctions of the same tenor longer than this
# is flagged for a human to look at -- it is not automatically an
# error (issuance calendars do change), just a candidate to review.
SUSPICIOUS_GAP_DAYS = 150


def _missingness(df: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, float]:
    present = [f for f in fields if f in df.columns]
    if not df.shape[0] or not present:
        return {}
    return {f: float(df[f].isna().mean()) for f in present}


def _coverage_by_year(nominal: pd.DataFrame) -> dict[int, int]:
    if nominal.empty:
        return {}
    years = nominal["auction_date"].dt.year
    return {int(y): int(c) for y, c in years.value_counts().sort_index().items()}


def _coverage_by_tenor(nominal: pd.DataFrame) -> dict[str, int]:
    counts = nominal["tenor"].value_counts()
    return {tenor: int(counts.get(tenor, 0)) for tenor in NOMINAL_COUPON_TENORS}


def _new_issue_vs_reopening_by_tenor(nominal: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for tenor in NOMINAL_COUPON_TENORS:
        subset = nominal.loc[nominal["tenor"] == tenor]
        result[tenor] = {
            "new_issue": int((~subset["is_reopening"]).sum()),
            "reopening": int(subset["is_reopening"].sum()),
        }
    return result


def _suspicious_gaps(nominal: pd.DataFrame) -> dict[str, dict[str, Any]]:
    gaps: dict[str, dict[str, Any]] = {}
    for tenor in NOMINAL_COUPON_TENORS:
        dates = nominal.loc[nominal["tenor"] == tenor, "auction_date"].dropna().sort_values()
        if len(dates) < 2:
            continue
        diffs = dates.diff().dropna().dt.days
        max_gap = int(diffs.max())
        if max_gap > SUSPICIOUS_GAP_DAYS:
            gap_end = dates.iloc[int(diffs.values.argmax()) + 1]
            gap_start = dates.iloc[int(diffs.values.argmax())]
            gaps[tenor] = {
                "max_gap_days": max_gap,
                "gap_start": gap_start.date().isoformat(),
                "gap_end": gap_end.date().isoformat(),
            }
    return gaps


def _excluded_records(df: pd.DataFrame) -> dict[str, Any]:
    excluded = df.loc[~df["is_nominal_coupon"]]
    bills = excluded.loc[excluded["security_type"] == "Bill"]
    tips = excluded.loc[excluded["inflation_index_security"] == "Yes"]
    frn = excluded.loc[excluded["floating_rate"] == "Yes"]
    accounted_for = len(bills) + len(tips) + len(frn)
    return {
        "total_excluded": len(excluded),
        "bills": len(bills),
        "tips": len(tips),
        "frn": len(frn),
        "unaccounted_for": int(len(excluded) - accounted_for),
    }


def _pending_auctions(nominal: pd.DataFrame) -> list[str]:
    pending = nominal.loc[~nominal["results_available"]]
    return sorted(d.isoformat() for d in pending["auction_date"].dt.date.dropna())


def build_quality_report(
    df: pd.DataFrame,
    anomalies: dict[str, Any],
    schema_drift: dict[str, list[str]],
) -> dict[str, Any]:
    nominal = nominal_coupon_subset(df)
    duplicates = find_duplicate_keys(df)
    settled_nominal = nominal.loc[nominal["results_available"]]

    return {
        "total_rows": len(df),
        "overall_date_range": (
            None
            if df["auction_date"].dropna().empty
            else [
                df["auction_date"].min().date().isoformat(),
                df["auction_date"].max().date().isoformat(),
            ]
        ),
        "nominal_coupon_row_count": len(nominal),
        "coverage_by_year": _coverage_by_year(nominal),
        "coverage_by_tenor": _coverage_by_tenor(nominal),
        "new_issue_vs_reopening_by_tenor": _new_issue_vs_reopening_by_tenor(nominal),
        "candidate_primary_key": list(CANDIDATE_PRIMARY_KEY),
        "duplicate_key_row_count": len(duplicates),
        "duplicate_keys": duplicates[list(CANDIDATE_PRIMARY_KEY)].to_dict("records")
        if not duplicates.empty
        else [],
        "missingness_full_dataset": _missingness(
            df, tuple(CANDIDATE_TARGET_FIELDS) + ("auction_date", "cusip")
        ),
        "missingness_settled_nominal_subset": _missingness(
            settled_nominal, CANDIDATE_TARGET_FIELDS
        ),
        "candidate_target_field_coverage_settled_nominal": {
            f: 1.0 - settled_nominal[f].isna().mean() if f in settled_nominal.columns and len(settled_nominal) else None
            for f in CANDIDATE_TARGET_FIELDS
        },
        "pending_or_unsettled_nominal_auctions": _pending_auctions(nominal),
        "unexpected_categories": anomalies.get("unexpected_categories", {}),
        "invalid_dates": anomalies.get("invalid_dates", {}),
        "invalid_numbers": anomalies.get("invalid_numbers", {}),
        "schema_drift": schema_drift,
        "suspicious_gaps_by_tenor": _suspicious_gaps(nominal),
        "excluded_records": _excluded_records(df),
    }


def render_quality_report_markdown(report: dict[str, Any], generated_note: str) -> str:
    lines: list[str] = []
    lines.append("# Treasury Auction Data Quality Report")
    lines.append("")
    lines.append(generated_note)
    lines.append("")
    lines.append(
        "Every figure below was computed by running the pipeline in "
        "`src/treasury_auction_stress/data/` against the actual downloaded "
        "dataset. None of it is hand-typed or estimated."
    )
    lines.append("")

    lines.append("## Overall dataset")
    lines.append(f"- Total rows retrieved (all security types): **{report['total_rows']}**")
    date_range = report["overall_date_range"]
    lines.append(
        f"- Overall auction-date range: **{date_range[0]} to {date_range[1]}**"
        if date_range
        else "- Overall auction-date range: no valid auction dates found"
    )
    lines.append(
        f"- Rows in the nominal-coupon analytical subset: "
        f"**{report['nominal_coupon_row_count']}**"
    )
    lines.append("")

    lines.append("## Coverage by year (nominal-coupon subset)")
    lines.append("| Year | Auctions |")
    lines.append("|---|---|")
    for year, count in report["coverage_by_year"].items():
        lines.append(f"| {year} | {count} |")
    lines.append("")

    lines.append("## Coverage by tenor (nominal-coupon subset)")
    lines.append("| Tenor | Auctions | New issues | Reopenings |")
    lines.append("|---|---|---|---|")
    for tenor, count in report["coverage_by_tenor"].items():
        split = report["new_issue_vs_reopening_by_tenor"].get(tenor, {})
        lines.append(
            f"| {tenor} | {count} | {split.get('new_issue', 0)} | {split.get('reopening', 0)} |"
        )
    lines.append("")

    lines.append("## Candidate primary key and duplicates")
    lines.append(f"- Candidate primary key: `{tuple(report['candidate_primary_key'])}`")
    lines.append(
        f"- Rows sharing a duplicate key: **{report['duplicate_key_row_count']}**"
        + (" (none found)" if report["duplicate_key_row_count"] == 0 else "")
    )
    if report["duplicate_keys"]:
        lines.append("")
        lines.append("Duplicate keys found:")
        for row in report["duplicate_keys"]:
            lines.append(f"- {row}")
    lines.append("")

    lines.append("## Missingness")
    lines.append("Full downloaded dataset (all security types, includes Bills/TIPS/FRN, "
                  "which are not expected to populate every nominal-coupon-oriented field):")
    lines.append("")
    lines.append("| Field | Fraction missing |")
    lines.append("|---|---|")
    for field, frac in report["missingness_full_dataset"].items():
        lines.append(f"| {field} | {frac:.3f} |")
    lines.append("")
    lines.append(
        "Nominal-coupon subset, restricted to auctions whose results are "
        "already published (excludes pending/future auctions, listed below):"
    )
    lines.append("")
    lines.append("| Field | Fraction missing |")
    lines.append("|---|---|")
    for field, frac in report["missingness_settled_nominal_subset"].items():
        lines.append(f"| {field} | {frac:.3f} |")
    lines.append("")

    lines.append("## Candidate target-field coverage (settled nominal-coupon subset)")
    lines.append("| Field | Coverage (non-null fraction) |")
    lines.append("|---|---|")
    for field, cov in report["candidate_target_field_coverage_settled_nominal"].items():
        lines.append(f"| {field} | {'n/a' if cov is None else f'{cov:.3f}'} |")
    lines.append("")

    pending = report["pending_or_unsettled_nominal_auctions"]
    lines.append(f"## Pending / unsettled nominal-coupon auctions ({len(pending)})")
    lines.append(
        "These are auctions that have been announced (and may have already "
        "occurred) but whose result fields (bid-to-cover, dealer allotment, "
        "etc.) are still `null` in the source data as of retrieval time -- "
        "this is expected for auctions on or shortly before the retrieval "
        "date, not a data-quality defect. They are excluded from the "
        "missingness and coverage figures above."
    )
    if pending:
        lines.append("")
        for d in pending:
            lines.append(f"- {d}")
    lines.append("")

    lines.append("## Schema anomalies")
    lines.append(f"- Unexpected category values: {report['unexpected_categories'] or 'none found'}")
    lines.append(f"- Invalid (unparseable) date values by field: {report['invalid_dates'] or 'none found'}")
    lines.append(f"- Invalid (unparseable) numeric values by field: {report['invalid_numbers'] or 'none found'}")
    drift = report["schema_drift"]
    lines.append(
        f"- Columns present now but not in the schema captured on 2026-09-10: "
        f"{drift.get('new_columns') or 'none'}"
    )
    lines.append(
        f"- Columns expected (from the 2026-09-10 schema capture) but missing now: "
        f"{drift.get('missing_columns') or 'none'}"
    )
    lines.append("")

    lines.append("## Suspicious gaps between consecutive auctions of the same tenor")
    gaps = report["suspicious_gaps_by_tenor"]
    if not gaps:
        lines.append(f"None found above the {SUSPICIOUS_GAP_DAYS}-day threshold.")
    else:
        lines.append(f"Gaps longer than {SUSPICIOUS_GAP_DAYS} days (flagged for review, not removed):")
        lines.append("")
        for tenor, info in gaps.items():
            lines.append(
                f"- {tenor}: {info['max_gap_days']} days, between "
                f"{info['gap_start']} and {info['gap_end']}"
            )
    lines.append("")

    lines.append("## Records excluded from the nominal-coupon analytical subset")
    excl = report["excluded_records"]
    lines.append(f"- Total excluded: **{excl['total_excluded']}**")
    lines.append(f"  - Bills: {excl['bills']}")
    lines.append(f"  - TIPS (inflation-indexed): {excl['tips']}")
    lines.append(f"  - Floating-rate notes: {excl['frn']}")
    lines.append(
        f"  - Unaccounted for (should be 0 -- would indicate an unrecognized "
        f"security type): {excl['unaccounted_for']}"
    )
    lines.append("")

    return "\n".join(lines)
