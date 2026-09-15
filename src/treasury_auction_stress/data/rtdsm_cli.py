"""Phase 4C entry point: download, normalize, and write the processed
Philadelphia Fed RTDSM macro-vintage tables.

Never uses FRED/ALFRED for anything, including gap-filling -- see
`docs/data_source_governance.md`.

Run it via uv, from the repository root:

    uv run python -m treasury_auction_stress.data.rtdsm_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from treasury_auction_stress.data.rtdsm_client import RtdsmApiError
from treasury_auction_stress.data.rtdsm_download import download_rtdsm
from treasury_auction_stress.data.rtdsm_normalize import (
    attach_release_dates,
    build_snapshot_table,
    parse_workbook,
    vintage_index,
)
from treasury_auction_stress.data.rtdsm_raw_store import load_raw_bytes
from treasury_auction_stress.data.rtdsm_schema import VARIABLE_BY_MNEMONIC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/rtdsm", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Downloading Philadelphia Fed RTDSM (6 selected macro variables, never FRED/ALFRED) ...")
    try:
        results = download_rtdsm(raw_dir=args.raw_dir, force_refresh=args.force_refresh)
    except RtdsmApiError as exc:
        print(f"ERROR: could not reach the Philadelphia Fed RTDSM endpoint: {exc}", file=sys.stderr)
        print("No fabricated or substitute data will be produced.", file=sys.stderr)
        return 1

    n_cached = sum(1 for r in results if r.was_cached)
    print(f"OK: {len(results)} variable(s) processed ({n_cached} reused from cache, {len(results) - n_cached} freshly downloaded)")

    long_frames, vintage_frames, snapshot_frames = [], [], []
    all_anomalies: dict[str, dict] = {}
    for result in results:
        mnemonic = result.metadata.mnemonic
        variable = VARIABLE_BY_MNEMONIC[mnemonic]
        raw_bytes = load_raw_bytes(result.raw_path)
        long_df, anomalies = parse_workbook(raw_bytes, variable)
        if anomalies["unrecognized_vintage_columns"] or anomalies["unparseable_observation_periods"]:
            all_anomalies[mnemonic] = anomalies
        long_df = attach_release_dates(long_df, variable)

        long_frames.append(long_df)
        vintage_frames.append(vintage_index(long_df).assign(mnemonic=mnemonic))
        snapshot_frames.append(build_snapshot_table(long_df, variable).assign(mnemonic=mnemonic))

    if all_anomalies:
        print(f"  WARNING: schema anomalies detected: {all_anomalies}")

    combined_long = pd.concat(long_frames, ignore_index=True)
    combined_vintages = pd.concat(vintage_frames, ignore_index=True)
    combined_snapshots = pd.concat(snapshot_frames, ignore_index=True)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.processed_dir / "rtdsm_long.parquet"
    vintages_path = args.processed_dir / "rtdsm_vintage_index.parquet"
    snapshots_path = args.processed_dir / "rtdsm_snapshots.parquet"
    combined_long.to_parquet(long_path, index=False)
    combined_vintages.to_parquet(vintages_path, index=False)
    combined_snapshots.to_parquet(snapshots_path, index=False)
    print(f"  wrote tidy long table: {long_path} ({len(combined_long)} rows)")
    print(f"  wrote vintage index: {vintages_path} ({len(combined_vintages)} rows)")
    print(f"  wrote snapshot table: {snapshots_path} ({len(combined_snapshots)} rows)")
    for mnemonic in VARIABLE_BY_MNEMONIC:
        sub = combined_long[combined_long["mnemonic"] == mnemonic]
        print(f"  {mnemonic}: {sub['observation_date'].min().date()} to {sub['observation_date'].max().date()} observations, {sub['vintage_label'].nunique()} vintages")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
