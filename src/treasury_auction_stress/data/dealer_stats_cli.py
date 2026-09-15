"""Command-line entry point for Phase 3: download, normalize, and
report on NY Fed Primary Dealer Statistics.

Run it via uv, from the repository root:

    uv run python -m treasury_auction_stress.data.dealer_stats_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from treasury_auction_stress.data.dealer_stats_client import DealerStatsApiError
from treasury_auction_stress.data.dealer_stats_download import download_dealer_stats
from treasury_auction_stress.data.dealer_stats_normalize import (
    check_schema_drift,
    normalize_dealer_stats,
    pivot_wide,
)
from treasury_auction_stress.data.dealer_stats_raw_store import load_raw_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download (or reuse a cached copy of) NY Fed Primary Dealer "
            "Statistics for this project's selected series, normalize "
            "them, and write the processed tables."
        )
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download even if a cached raw artifact for today (UTC retrieval date) already exists.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Downloading NY Fed Primary Dealer Statistics (selected series) ...")
    try:
        result = download_dealer_stats(raw_dir=args.raw_dir, force_refresh=args.force_refresh)
    except DealerStatsApiError as exc:
        print(f"ERROR: could not reach the NY Fed Primary Dealer Statistics API: {exc}", file=sys.stderr)
        print(
            "No fabricated or substitute data will be produced. "
            "See docs/point_in_time_rules.md.",
            file=sys.stderr,
        )
        return 1

    cache_note = "reused cached artifact from" if result.was_cached else "downloaded fresh artifact for"
    print(f"OK: {cache_note} retrieval date {result.retrieval_date} (UTC calendar date)")
    print(f"  raw file: {result.raw_path}")
    print(f"  rows: {result.metadata.row_count}")
    print(f"  date range: {result.metadata.dataset_start_date} to {result.metadata.dataset_end_date}")

    payload = load_raw_payload(result.raw_path)
    series_list = payload.get("series_list")
    drift = check_schema_drift(series_list.get("body") if series_list else None)
    if drift.get("missing_keyids"):
        print(f"  WARNING: selected keyids no longer listed as active: {drift['missing_keyids']}")

    long_df, anomalies = normalize_dealer_stats(payload)
    if anomalies.get("unselected_keyids"):
        print(f"  WARNING: raw payload contained unselected keyids: {anomalies['unselected_keyids']}")
    wide_df = pivot_wide(long_df)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.processed_dir / "dealer_stats_long.parquet"
    wide_path = args.processed_dir / "dealer_stats_wide.parquet"
    long_df.to_parquet(long_path, index=False)
    wide_df.to_parquet(wide_path, index=False)
    print(f"  wrote tidy long table: {long_path} ({len(long_df)} rows)")
    print(f"  wrote wide table: {wide_path} ({len(wide_df)} rows)")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
