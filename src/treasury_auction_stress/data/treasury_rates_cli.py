"""Phase 4A entry point: download, normalize, and write the processed
Treasury Daily Par Yield Curve tables.

Run it via uv, from the repository root:

    uv run python -m treasury_auction_stress.data.treasury_rates_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.treasury_rates_client import TreasuryRatesApiError
from treasury_auction_stress.data.treasury_rates_download import download_treasury_rates
from treasury_auction_stress.data.treasury_rates_normalize import (
    parse_csv_to_wide,
    pivot_wide_by_maturity,
    to_long,
)
from treasury_auction_stress.data.treasury_rates_raw_store import load_raw_csv_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Downloading Treasury Daily Par Yield Curve rates (2009-present, one request per year) ...")
    try:
        results = download_treasury_rates(raw_dir=args.raw_dir, force_refresh=args.force_refresh)
    except TreasuryRatesApiError as exc:
        print(f"ERROR: could not reach the Treasury interest-rates endpoint: {exc}", file=sys.stderr)
        print("No fabricated or substitute data will be produced.", file=sys.stderr)
        return 1

    n_cached = sum(1 for r in results if r.was_cached)
    print(f"OK: {len(results)} year(s) processed ({n_cached} reused from cache, {len(results) - n_cached} freshly downloaded)")

    all_anomalies: dict[int, dict] = {}
    long_frames = []
    for result in results:
        raw_text = load_raw_csv_text(result.raw_path)
        wide_df, anomalies = parse_csv_to_wide(raw_text, source_year=result.year)
        if anomalies.get("unrecognized_columns") or anomalies.get("invalid_dates"):
            all_anomalies[result.year] = anomalies
        raw_artifact_id = result.raw_path.name
        long_df = to_long(
            wide_df,
            retrieval_timestamp_utc=result.metadata.retrieval_timestamp_utc,
            raw_artifact_id=raw_artifact_id,
        )
        long_frames.append(long_df)

    if all_anomalies:
        print(f"  WARNING: schema anomalies detected: {all_anomalies}")

    combined_long = pd.concat(long_frames, ignore_index=True).sort_values(["maturity_label", "rate_date"]).reset_index(drop=True)
    wide_by_maturity = pivot_wide_by_maturity(combined_long)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.processed_dir / "treasury_rates_long.parquet"
    wide_path = args.processed_dir / "treasury_rates_wide.parquet"
    combined_long.to_parquet(long_path, index=False)
    wide_by_maturity.to_parquet(wide_path, index=False)
    print(f"  wrote tidy long table: {long_path} ({len(combined_long)} rows)")
    print(f"  wrote wide table: {wide_path} ({len(wide_by_maturity)} rows)")
    print(f"  date range: {combined_long['rate_date'].min().date()} to {combined_long['rate_date'].max().date()}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
