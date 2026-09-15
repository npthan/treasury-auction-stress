"""Phase 4B entry point: download, normalize, and write the processed
CFTC TFF Futures Only positioning tables.

Run it via uv, from the repository root:

    uv run python -m treasury_auction_stress.data.cftc_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from treasury_auction_stress.data.cftc_client import CftcApiError
from treasury_auction_stress.data.cftc_download import download_cftc_positioning
from treasury_auction_stress.data.cftc_normalize import (
    normalize_cftc_positioning,
    pivot_wide_by_contract,
)
from treasury_auction_stress.data.cftc_raw_store import load_raw_payload
from treasury_auction_stress.data.cftc_schema import CFTC_START_DATE
from treasury_auction_stress.features.cftc_candidate_features import (
    build_positioning_feature_table,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/cftc", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Downloading CFTC TFF Futures Only positioning (6 selected Treasury-futures contracts) ...")
    try:
        result = download_cftc_positioning(raw_dir=args.raw_dir, force_refresh=args.force_refresh)
    except CftcApiError as exc:
        print(f"ERROR: could not reach the CFTC Socrata endpoint: {exc}", file=sys.stderr)
        print("No fabricated or substitute data will be produced.", file=sys.stderr)
        return 1

    print(f"OK: {'reused cached' if result.was_cached else 'freshly downloaded'} raw artifact ({result.metadata.row_count} raw rows)")

    payload = load_raw_payload(result.raw_path)
    long_df, anomalies = normalize_cftc_positioning(payload, retrieval_timestamp_utc=result.metadata.retrieval_timestamp_utc)
    if anomalies["unrecognized_contract_codes"] or anomalies["duplicate_report_rows"]:
        print(f"  WARNING: anomalies detected: {anomalies}")

    # Phase 4 acceptance review, issue 8: this used to truncate at
    # CFTC_START_DATE (2010-01-01, the auction modeling sample's own
    # start date), which created *artificial* missingness for early-2010
    # auctions whose as-of cutoff fell before the first >=2010 report was
    # safely available. The CFTC's own Socrata endpoint already serves
    # this project's selected contracts back to 2006-06-13 (verified),
    # so the full downloaded history is retained here -- the auction
    # modeling sample's own start date is unchanged; only the source
    # lookback used to construct legitimate point-in-time features is
    # extended, per the review's explicit instruction.
    long_df = long_df.reset_index(drop=True)
    feat_df = build_positioning_feature_table(long_df)
    wide_df = pivot_wide_by_contract(feat_df)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.processed_dir / "cftc_positioning_long.parquet"
    wide_path = args.processed_dir / "cftc_positioning_wide.parquet"
    feat_df.to_parquet(long_path, index=False)
    wide_df.to_parquet(wide_path, index=False)
    print(f"  wrote tidy long table: {long_path} ({len(feat_df)} rows)")
    print(f"  wrote wide table: {wide_path} ({len(wide_df)} rows)")
    print(f"  date range (full retained source history, not truncated to the {CFTC_START_DATE} modeling-sample start): "
          f"{long_df['report_date'].min().date()} to {long_df['report_date'].max().date()}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
