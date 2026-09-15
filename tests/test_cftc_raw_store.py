from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from treasury_auction_stress.data.cftc_client import FetchedContract
from treasury_auction_stress.data.cftc_raw_store import (
    RawArtifactConflictError,
    is_cached,
    load_metadata,
    load_raw_payload,
    raw_artifact_paths,
    write_raw_artifact,
)

_BODY = [
    {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "open_interest_all": "4500000"},
    {"report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000", "open_interest_all": "4600000"},
]


def _contracts(body=_BODY) -> list[FetchedContract]:
    return [
        FetchedContract(
            contract_code="043602", url="https://example.invalid/043602", http_status=200,
            retrieved_at_utc="2026-09-11T00:00:00+00:00", raw_text=json.dumps(body),
        )
    ]


def test_write_raw_artifact_creates_files_and_correct_metadata(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2026-09-11")
    assert raw_path.exists() and meta_path.exists()
    assert metadata.row_count == 2
    assert metadata.dataset_date_range == ["2024-01-02T00:00:00.000", "2024-01-09T00:00:00.000"]
    assert metadata.source_name == "cftc_tff_futures_only"
    assert metadata.contract_codes == ["043602"]


def test_checksum_matches_file_contents(tmp_path: Path):
    metadata = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    actual = hashlib.sha256(raw_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert actual == metadata.checksum_sha256


def test_repeat_write_is_idempotent(tmp_path: Path):
    first = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    mtime = raw_path.stat().st_mtime_ns
    second = write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    assert second.checksum_sha256 == first.checksum_sha256
    assert raw_path.stat().st_mtime_ns == mtime


def test_raw_artifact_is_immutable(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    with pytest.raises(RawArtifactConflictError):
        write_raw_artifact(
            tmp_path, retrieval_date="2026-09-11",
            contracts=_contracts([{"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "open_interest_all": "999"}]),
        )


def test_is_cached_true_after_write_false_before(tmp_path: Path):
    assert is_cached(tmp_path, "2026-09-11") is False
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    assert is_cached(tmp_path, "2026-09-11") is True


def test_is_cached_false_if_corrupted(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    raw_path.write_text("corrupted", encoding="utf-8")
    assert is_cached(tmp_path, "2026-09-11") is False


def test_load_raw_payload_round_trips(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, meta_path = raw_artifact_paths(tmp_path, "2026-09-11")
    payload = load_raw_payload(raw_path)
    assert payload["contracts"][0]["body"] == _BODY
    metadata = load_metadata(meta_path)
    assert metadata.contract_codes == ["043602"]


def test_no_leftover_tmp_file_after_write(tmp_path: Path):
    write_raw_artifact(tmp_path, retrieval_date="2026-09-11", contracts=_contracts())
    raw_path, _ = raw_artifact_paths(tmp_path, "2026-09-11")
    tmp_marker = raw_path.with_suffix(raw_path.suffix + ".tmp")
    assert not tmp_marker.exists()
