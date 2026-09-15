"""Persisting raw Treasury Daily Par Yield Curve CSV responses to disk
as immutable, checksummed artifacts -- one artifact per calendar year,
mirroring the source's own natural pagination unit. Same
immutability/idempotence/checksum contract as
`treasury_auction_stress.data.raw_store` and `.dealer_stats_raw_store`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treasury_auction_stress.data.treasury_rates_client import FetchedYear

RAW_ARTIFACT_FORMAT_VERSION = "1"
SOURCE_NAME = "treasury_daily_par_yield_curve"


class RawArtifactConflictError(Exception):
    """A raw artifact already exists with *different* content than
    what we are about to write for the same (year, retrieval date).
    """


@dataclass(frozen=True)
class RawArtifactMetadata:
    source_name: str
    year: int
    request_url: str
    retrieval_timestamp_utc: str
    http_status: int
    content_type: str
    response_format: str
    row_count: int
    dataset_date_range: list[str]
    checksum_sha256: str
    raw_artifact_format_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_artifact_paths(raw_dir: Path, year: int, retrieval_date: str) -> tuple[Path, Path]:
    stem = f"treasury_par_yield_curve_raw_{year}_retrieved_{retrieval_date}"
    return raw_dir / f"{stem}.csv", raw_dir / f"{stem}.meta.json"


def _row_count_and_date_range(raw_text: str) -> tuple[int, list[str]]:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0, []
    data_lines = lines[1:]
    dates = [line.split(",")[0].strip('"') for line in data_lines]
    return len(data_lines), [dates[-1], dates[0]] if dates else []


def is_cached(raw_dir: Path, year: int, retrieval_date: str) -> bool:
    raw_path, meta_path = raw_artifact_paths(raw_dir, year, retrieval_date)
    if not raw_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = load_metadata(meta_path)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    actual_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
    return actual_checksum == metadata.checksum_sha256


def write_raw_artifact(raw_dir: Path, *, year: int, retrieval_date: str, fetched: FetchedYear) -> RawArtifactMetadata:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = raw_artifact_paths(raw_dir, year, retrieval_date)

    checksum = _sha256_of_text(fetched.raw_text)
    if raw_path.exists():
        existing_checksum = _sha256_of_text(raw_path.read_text(encoding="utf-8"))
        if existing_checksum == checksum:
            return load_metadata(meta_path)
        raise RawArtifactConflictError(
            f"{raw_path} already exists with different content "
            f"(existing checksum {existing_checksum[:12]}..., new checksum {checksum[:12]}...). "
            "Raw artifacts are immutable."
        )

    row_count, date_range = _row_count_and_date_range(fetched.raw_text)
    metadata = RawArtifactMetadata(
        source_name=SOURCE_NAME,
        year=year,
        request_url=fetched.url,
        retrieval_timestamp_utc=fetched.retrieved_at_utc,
        http_status=fetched.http_status,
        content_type=fetched.content_type,
        response_format="csv",
        row_count=row_count,
        dataset_date_range=date_range,
        checksum_sha256=checksum,
        raw_artifact_format_version=RAW_ARTIFACT_FORMAT_VERSION,
    )
    # Atomic final write: write to a temp file in the same directory, then
    # rename -- avoids a torn/partial raw artifact if interrupted mid-write.
    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    tmp_path.write_text(fetched.raw_text, encoding="utf-8")
    tmp_path.replace(raw_path)
    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


def find_snapshot_paths(raw_dir: Path, year: int) -> list[Path]:
    """Every immutable raw CSV artifact this project has ever retrieved
    for `year`, oldest retrieval date first -- for genuine
    revision-detection (Phase 4 acceptance review, issue 11), never a
    claim about revisions inferred merely from a single snapshot.
    """
    return sorted(raw_dir.glob(f"treasury_par_yield_curve_raw_{year}_retrieved_*.csv"))


def compare_snapshots(older_path: Path, newer_path: Path) -> dict[str, Any]:
    """Row-for-row, cell-for-cell comparison of two immutable raw
    snapshots of the same year. Returns a real, checked result -- never
    an inference from row count alone. Requires pandas at call time
    (imported locally to keep this module's base import light).
    """
    import pandas as pd

    older = pd.read_csv(older_path, dtype=str, keep_default_na=False)
    newer = pd.read_csv(newer_path, dtype=str, keep_default_na=False)

    older_dates = set(older["Date"]) if "Date" in older.columns else set()
    newer_dates = set(newer["Date"]) if "Date" in newer.columns else set()
    common = sorted(older_dates & newer_dates)

    changed_dates: list[str] = []
    if common:
        older_common = older[older["Date"].isin(common)].sort_values("Date").reset_index(drop=True)
        newer_common = newer[newer["Date"].isin(common)].sort_values("Date").reset_index(drop=True)
        shared_cols = [c for c in older_common.columns if c in newer_common.columns]
        diff_mask = (older_common[shared_cols] != newer_common[shared_cols]).any(axis=1)
        changed_dates = older_common.loc[diff_mask, "Date"].tolist()

    return {
        "older_path": str(older_path),
        "newer_path": str(newer_path),
        "dates_only_in_older": sorted(older_dates - newer_dates),
        "dates_only_in_newer": sorted(newer_dates - older_dates),
        "n_common_dates": len(common),
        "n_changed_dates": len(changed_dates),
        "changed_dates": changed_dates,
        "identical": not changed_dates and older_dates == newer_dates,
    }


def load_metadata(meta_path: Path) -> RawArtifactMetadata:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return RawArtifactMetadata(**data)


def load_raw_csv_text(raw_path: Path) -> str:
    return raw_path.read_text(encoding="utf-8")
