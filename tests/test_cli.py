"""Tests for the `treasury_auction_stress.data.cli` command.

The network call is monkeypatched out entirely -- this only tests that
the CLI correctly wires download -> normalize -> processed output ->
quality report, using a fixture-derived raw artifact.
"""

from __future__ import annotations

from pathlib import Path

from treasury_auction_stress.data import cli as cli_module
from treasury_auction_stress.data.client import FetchedPage
from treasury_auction_stress.data.download import DownloadResult
from treasury_auction_stress.data.raw_store import (
    raw_artifact_paths,
    write_raw_artifact,
)


def test_run_writes_processed_files_and_quality_report(
    tmp_path: Path, monkeypatch, sample_page_body
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    page = FetchedPage(
        url="https://example.invalid/?page=1",
        params={},
        http_status=200,
        retrieved_at_utc="2026-09-10T00:00:00+00:00",
        raw_text=str(sample_page_body),
        parsed=sample_page_body,
    )
    metadata = write_raw_artifact(
        raw_dir,
        start_date="2010-01-01",
        end_date="2026-09-10",
        retrieval_date="2026-09-10",
        pages=[page],
    )
    raw_path, meta_path = raw_artifact_paths(raw_dir, "2010-01-01", "2026-09-10", "2026-09-10")

    def fake_download_auctions(**kwargs):
        return DownloadResult(
            metadata=metadata,
            raw_path=raw_path,
            meta_path=meta_path,
            retrieval_date="2026-09-10",
            was_cached=False,
        )

    monkeypatch.setattr(cli_module, "download_auctions", fake_download_auctions)

    exit_code = cli_module.run(
        [
            "--start-date",
            "2010-01-01",
            "--end-date",
            "2026-09-10",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    assert exit_code == 0
    assert (processed_dir / "treasury_auctions_full.parquet").exists()
    assert (processed_dir / "treasury_auctions_nominal_coupons.parquet").exists()
    assert (reports_dir / "auction_data_quality.md").exists()
