from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from treasury_auction_stress.data.treasury_rates_client import FetchedYear
from treasury_auction_stress.data.treasury_rates_raw_store import (
    RawArtifactConflictError,
    compare_snapshots,
    find_snapshot_paths,
    is_cached,
    load_metadata,
    load_raw_csv_text,
    raw_artifact_paths,
    write_raw_artifact,
)

_CSV = 'Date,"1 Mo","2 Yr"\n01/03/2024,5.50,4.30\n01/02/2024,5.49,4.33\n'


def _fetched(text: str = _CSV) -> FetchedYear:
    return FetchedYear(
        year=2024, url="https://example.invalid/2024", http_status=200,
        content_type="text/csv; charset=UTF-8", retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_text=text,
    )


def test_write_raw_artifact_creates_files_and_correct_metadata(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, meta_path = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    assert raw_path.exists() and meta_path.exists()
    assert metadata.row_count == 2
    assert metadata.dataset_date_range == ["01/02/2024", "01/03/2024"]
    assert metadata.source_name == "treasury_daily_par_yield_curve"


def test_checksum_matches_file_contents(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    actual = hashlib.sha256(raw_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert actual == metadata.checksum_sha256


def test_repeat_write_is_idempotent(tmp_path: Path):
    first = write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    mtime = raw_path.stat().st_mtime_ns
    second = write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    assert second.checksum_sha256 == first.checksum_sha256
    assert raw_path.stat().st_mtime_ns == mtime


def test_raw_artifact_is_immutable(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    with pytest.raises(RawArtifactConflictError):
        write_raw_artifact(
            tmp_path, year=2024, retrieval_date="2026-09-11",
            fetched=_fetched('Date,"1 Mo"\n01/03/2024,9.99\n'),
        )


def test_is_cached_true_after_write_false_before(tmp_path: Path):
    assert is_cached(tmp_path, 2024, "2026-09-11") is False
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    assert is_cached(tmp_path, 2024, "2026-09-11") is True


def test_is_cached_false_if_corrupted(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    raw_path.write_text("corrupted", encoding="utf-8")
    assert is_cached(tmp_path, 2024, "2026-09-11") is False


def test_load_raw_csv_text_round_trips(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, meta_path = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    assert load_raw_csv_text(raw_path) == _CSV
    metadata = load_metadata(meta_path)
    assert metadata.year == 2024


def test_no_leftover_tmp_file_after_write(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    raw_path, _ = raw_artifact_paths(tmp_path, 2024, "2026-09-11")
    tmp_marker = raw_path.with_suffix(raw_path.suffix + ".tmp")
    assert not tmp_marker.exists()


# -- Phase 4 acceptance review, issue 11: genuine multi-snapshot revision detection --


def test_find_snapshot_paths_returns_all_retrieval_dates_oldest_first(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-10", fetched=_fetched())
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-12", fetched=_fetched())
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    paths = find_snapshot_paths(tmp_path, 2024)
    assert [p.name for p in paths] == [
        "treasury_par_yield_curve_raw_2024_retrieved_2026-09-10.csv",
        "treasury_par_yield_curve_raw_2024_retrieved_2026-09-11.csv",
        "treasury_par_yield_curve_raw_2024_retrieved_2026-09-12.csv",
    ]


def test_compare_snapshots_detects_no_change_between_identical_snapshots(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-12", fetched=_fetched())
    paths = find_snapshot_paths(tmp_path, 2024)
    result = compare_snapshots(paths[0], paths[1])
    assert result["identical"] is True
    assert result["n_changed_dates"] == 0


def test_compare_snapshots_detects_a_real_value_change(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    revised_csv = 'Date,"1 Mo","2 Yr"\n01/03/2024,5.50,4.30\n01/02/2024,5.49,9.99\n'  # 2 Yr corrected
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-12", fetched=_fetched(revised_csv))
    paths = find_snapshot_paths(tmp_path, 2024)
    result = compare_snapshots(paths[0], paths[1])
    assert result["identical"] is False
    assert result["n_changed_dates"] == 1
    assert "01/02/2024" in result["changed_dates"]


def test_compare_snapshots_detects_added_or_removed_dates(tmp_path: Path):
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-11", fetched=_fetched())
    extended_csv = 'Date,"1 Mo","2 Yr"\n01/04/2024,5.51,4.31\n01/03/2024,5.50,4.30\n01/02/2024,5.49,4.33\n'
    write_raw_artifact(tmp_path, year=2024, retrieval_date="2026-09-12", fetched=_fetched(extended_csv))
    paths = find_snapshot_paths(tmp_path, 2024)
    result = compare_snapshots(paths[0], paths[1])
    assert result["dates_only_in_newer"] == ["01/04/2024"]
    assert result["dates_only_in_older"] == []
