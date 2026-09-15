from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from treasury_auction_stress.features import feature_matrix_cli
from treasury_auction_stress.features.feature_matrix_cli import (
    PROCESSED_OUTPUT_FILES,
    REQUIRED_PROCESSED_FILES,
)


def _real_processed_dir_or_skip() -> Path:
    processed = Path("data/processed")
    if not all((processed / f).exists() for f in REQUIRED_PROCESSED_FILES):
        pytest.skip("processed Phase 1/3/4 tables not present in this environment")
    return processed


@pytest.fixture()
def isolated_processed_dir(tmp_path: Path) -> Path:
    """A copy of the real prerequisite processed files in an isolated
    temp directory, so a full `feature_matrix_cli.run` can read/write
    without touching the actual project's `data/processed/` or
    `artifacts/` during a test.
    """
    real_dir = _real_processed_dir_or_skip()
    isolated = tmp_path / "processed"
    isolated.mkdir()
    for filename in REQUIRED_PROCESSED_FILES:
        shutil.copy2(real_dir / filename, isolated / filename)
    return isolated


def test_run_writes_all_six_outputs_and_three_reports_on_success(isolated_processed_dir, tmp_path):
    reports_dir = tmp_path / "reports"
    exit_code = feature_matrix_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)]
    )
    assert exit_code == 0
    for filename in PROCESSED_OUTPUT_FILES:
        assert (isolated_processed_dir / filename).exists(), f"missing {filename}"
    for filename in ("phase_5_feature_matrix.md", "phase_5_leakage_audit.md", "phase_5_feature_dictionary.md"):
        assert (reports_dir / filename).exists(), f"missing {filename}"


def test_a_failing_gate_leaves_previously_published_outputs_untouched(isolated_processed_dir, tmp_path, monkeypatch):
    """Issue 3 (fail before writing): seed a "previously published" set
    of outputs (deliberately just placeholder content -- this test is
    about whether a failing run touches them at all, not about their
    own validity), force `run_all_gates` to fail, and confirm every
    output file (and every report) is byte-for-byte unchanged -- a
    failing audit must never replace previously valid outputs.

    Deliberately does NOT run the full pipeline first (that is already
    covered, for real, by `test_run_writes_all_six_outputs_and_three_
    reports_on_success` above) -- seeding placeholder files instead
    keeps this test fast while still proving the actual property of
    interest: a failing `run_all_gates` call must not write anything.
    """
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    before = {}
    for filename in PROCESSED_OUTPUT_FILES:
        path = isolated_processed_dir / filename
        path.write_bytes(f"placeholder content for {filename}".encode())
        before[filename] = path.read_bytes()
    for filename in ("phase_5_feature_matrix.md", "phase_5_leakage_audit.md", "phase_5_feature_dictionary.md"):
        path = reports_dir / filename
        path.write_text(f"placeholder content for {filename}")
        before[filename] = path.read_bytes()

    def _always_fail(artifacts, tables):
        raise feature_matrix_cli.Phase5GenerationError("deliberately induced failure for the acceptance-review test")

    monkeypatch.setattr(feature_matrix_cli, "run_all_gates", _always_fail)

    exit_code = feature_matrix_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)]
    )
    assert exit_code == 1

    for filename in PROCESSED_OUTPUT_FILES:
        assert (isolated_processed_dir / filename).read_bytes() == before[filename], f"{filename} was modified by a failing run"
    for filename in ("phase_5_feature_matrix.md", "phase_5_leakage_audit.md", "phase_5_feature_dictionary.md"):
        assert (reports_dir / filename).read_bytes() == before[filename], f"{filename} was modified by a failing run"

    # No stray temporary files left behind either.
    leftover_tmp = list(isolated_processed_dir.glob(".*.tmp")) + list(reports_dir.glob(".*.tmp"))
    assert leftover_tmp == [], f"temp files leaked: {leftover_tmp}"


def test_a_failing_structural_validation_writes_nothing_on_a_fresh_directory(isolated_processed_dir, tmp_path, monkeypatch):
    """Same discipline, but for a failure in `build_all_artifacts`
    (structural validation) rather than `run_all_gates`, and starting
    from a directory with NO prior outputs at all -- must still write
    nothing.
    """
    reports_dir = tmp_path / "reports"

    def _always_fail(nominal_df, tables):
        raise ValueError("deliberately induced structural failure")

    monkeypatch.setattr(feature_matrix_cli, "build_all_artifacts", _always_fail)

    exit_code = feature_matrix_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)]
    )
    assert exit_code == 1
    for filename in PROCESSED_OUTPUT_FILES:
        assert not (isolated_processed_dir / filename).exists()
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []


def test_atomic_write_parquet_does_not_touch_existing_file_on_failure(tmp_path, monkeypatch):
    import pandas as pd

    from treasury_auction_stress.features.feature_matrix_cli import (
        _atomic_write_parquet,
    )

    final_path = tmp_path / "table.parquet"
    good_df = pd.DataFrame({"a": [1, 2, 3]})
    _atomic_write_parquet(good_df, final_path)
    original_bytes = final_path.read_bytes()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    bad_df = pd.DataFrame({"a": [999]})
    with pytest.raises(RuntimeError):
        _atomic_write_parquet(bad_df, final_path)

    assert final_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_text_does_not_touch_existing_file_on_failure(tmp_path, monkeypatch):
    import os

    from treasury_auction_stress.features.feature_matrix_cli import _atomic_write_text

    final_path = tmp_path / "report.md"
    _atomic_write_text("original content", final_path)
    original = final_path.read_text()

    class _BoomHandle:
        def write(self, *args, **kwargs):
            raise RuntimeError("simulated write failure")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fdopen_boom(fd, *args, **kwargs):
        os.close(fd)
        return _BoomHandle()

    monkeypatch.setattr("treasury_auction_stress.features.feature_matrix_cli.os.fdopen", _fdopen_boom)

    with pytest.raises(RuntimeError):
        _atomic_write_text("new content", final_path)

    assert final_path.read_text() == original
    assert list(tmp_path.glob(".*.tmp")) == []
