from __future__ import annotations

from pathlib import Path

import pytest

from treasury_auction_stress.data.client import FetchedPage
from treasury_auction_stress.data.raw_store import (
    RawArtifactConflictError,
    is_cached,
    load_metadata,
    load_raw_payload,
    raw_artifact_paths,
    write_raw_artifact,
)


def _page(body: dict, url: str = "https://example.invalid/?page=1") -> FetchedPage:
    return FetchedPage(
        url=url,
        params={"page[number]": 1, "page[size]": 10, "sort": "auction_date"},
        http_status=200,
        retrieved_at_utc="2026-09-10T00:00:00+00:00",
        raw_text=str(body),
        parsed=body,
    )


def test_write_raw_artifact_creates_raw_file_and_sidecar(tmp_path: Path, sample_page_body):
    metadata = write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10")
    assert raw_path.exists()
    assert meta_path.exists()
    assert metadata.row_count == len(sample_page_body["data"])
    assert metadata.http_status_codes == [200]
    assert metadata.source_name == "treasury_fiscal_data_auctions"


def test_raw_artifact_checksum_matches_file_contents(tmp_path: Path, sample_page_body):
    metadata = write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    raw_path, _ = raw_artifact_paths(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10")
    import hashlib

    actual = hashlib.sha256(raw_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert actual == metadata.checksum_sha256


def test_repeat_run_with_identical_data_is_idempotent(tmp_path: Path, sample_page_body):
    kwargs = {
        "start_date": "2010-01-01",
        "end_date": "2026-09-10",
        "retrieval_date": "2026-09-10",
        "pages": [_page(sample_page_body)],
    }
    first = write_raw_artifact(tmp_path, **kwargs)
    raw_path, _ = raw_artifact_paths(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10")
    mtime_after_first_write = raw_path.stat().st_mtime_ns

    second = write_raw_artifact(tmp_path, **kwargs)

    assert second.checksum_sha256 == first.checksum_sha256
    # Rewriting identical content must not touch the file on disk again.
    assert raw_path.stat().st_mtime_ns == mtime_after_first_write


def test_raw_artifact_is_immutable_once_written(tmp_path: Path, sample_page_body, empty_page_body):
    write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    with pytest.raises(RawArtifactConflictError):
        write_raw_artifact(
            tmp_path,
            start_date="2010-01-01",
            end_date="2026-09-10",
            retrieval_date="2026-09-10",
            pages=[_page(empty_page_body)],  # different content, same identity
        )


def test_is_cached_true_after_write_and_false_before(tmp_path: Path, sample_page_body):
    assert is_cached(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10") is False
    write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    assert is_cached(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10") is True


def test_is_cached_false_if_raw_file_corrupted(tmp_path: Path, sample_page_body):
    write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    raw_path, _ = raw_artifact_paths(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10")
    raw_path.write_text("corrupted", encoding="utf-8")
    assert is_cached(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10") is False


def test_load_raw_payload_round_trips(tmp_path: Path, sample_page_body):
    write_raw_artifact(
        tmp_path,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[_page(sample_page_body)],
    )
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2010-01-01", "2026-09-10", "2026-09-10")
    payload = load_raw_payload(raw_path)
    assert payload["pages"][0]["body"] == sample_page_body

    metadata = load_metadata(meta_path)
    assert metadata.dataset_start_date == "2010-01-01"
    assert metadata.dataset_end_date == "2026-09-10"
