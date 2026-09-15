from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from treasury_auction_stress.evaluation import phase7_cli
from treasury_auction_stress.evaluation.data_loading import REQUIRED_FILES
from treasury_auction_stress.evaluation.phase7_cli import (
    FIGURE_FILES,
    PROCESSED_OUTPUT_FILES,
    REPORT_FILES,
    Phase7GenerationError,
)


def _real_processed_dir_or_skip() -> Path:
    processed = Path("data/processed")
    if not all((processed / f).exists() for f in REQUIRED_FILES):
        pytest.skip("processed Phase 5 tables not present in this environment")
    return processed


@pytest.fixture()
def isolated_processed_dir(tmp_path: Path) -> Path:
    """A copy of the real prerequisite Phase 5 files in an isolated temp
    directory, so a full `phase7_cli.run` can read/write without
    touching the actual project's `data/processed/` or `artifacts/`.
    """
    real_dir = _real_processed_dir_or_skip()
    isolated = tmp_path / "processed"
    isolated.mkdir()
    for filename in REQUIRED_FILES:
        shutil.copy2(real_dir / filename, isolated / filename)
    return isolated


def test_run_gate_tests_raises_on_a_real_failing_test_file(tmp_path):
    failing_test = tmp_path / "test_deliberately_failing.py"
    failing_test.write_text("def test_deliberately_failing():\n    assert False, 'induced failure'\n")

    with pytest.raises(Phase7GenerationError, match="failed"):
        phase7_cli._run_gate_tests(test_files=(str(failing_test),))


def test_run_gate_tests_passes_through_output_on_a_real_passing_test_file(tmp_path):
    passing_test = tmp_path / "test_deliberately_passing.py"
    passing_test.write_text("def test_deliberately_passing():\n    assert True\n")

    summary = phase7_cli._run_gate_tests(test_files=(str(passing_test),))
    assert "1 passed" in summary


def test_run_writes_all_outputs_on_success(isolated_processed_dir, tmp_path):
    reports_dir = tmp_path / "reports"
    exit_code = phase7_cli.run(["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)])
    assert exit_code == 0
    for filename in PROCESSED_OUTPUT_FILES:
        assert (isolated_processed_dir / filename).exists(), f"missing {filename}"
    for filename in REPORT_FILES:
        assert (reports_dir / filename).exists(), f"missing {filename}"
    for filename in FIGURE_FILES:
        assert (reports_dir / "figures" / filename).exists(), f"missing figure {filename}"


def test_a_failing_gate_leaves_previously_published_outputs_untouched(isolated_processed_dir, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True)

    before = {}
    for filename in PROCESSED_OUTPUT_FILES:
        path = isolated_processed_dir / filename
        path.write_bytes(f"placeholder content for {filename}".encode())
        before[filename] = path.read_bytes()
    for filename in REPORT_FILES:
        path = reports_dir / filename
        path.write_text(f"placeholder content for {filename}")
        before[filename] = path.read_bytes()
    for filename in FIGURE_FILES:
        path = figures_dir / filename
        path.write_bytes(f"placeholder figure bytes for {filename}".encode())
        before[filename] = path.read_bytes()

    def _always_fail(*, processed_dir, test_files=phase7_cli.DEFAULT_PHASE7_TEST_FILES):
        raise Phase7GenerationError("deliberately induced failure for the acceptance-review test")

    monkeypatch.setattr(phase7_cli, "_generate", _always_fail)

    exit_code = phase7_cli.run(["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)])
    assert exit_code == 1

    for filename in PROCESSED_OUTPUT_FILES:
        assert (isolated_processed_dir / filename).read_bytes() == before[filename], f"{filename} was modified by a failing run"
    for filename in REPORT_FILES:
        assert (reports_dir / filename).read_bytes() == before[filename], f"{filename} was modified by a failing run"
    for filename in FIGURE_FILES:
        assert (figures_dir / filename).read_bytes() == before[filename], f"{filename} was modified by a failing run"

    leftover_tmp = (
        list(isolated_processed_dir.glob(".*.tmp")) + list(reports_dir.glob(".*.tmp")) + list(figures_dir.glob(".*.tmp"))
    )
    assert leftover_tmp == [], f"temp files leaked: {leftover_tmp}"


def test_a_failing_gate_via_the_real_subprocess_mechanism_writes_nothing(isolated_processed_dir, tmp_path, monkeypatch):
    failing_test = tmp_path / "test_deliberately_failing.py"
    failing_test.write_text("def test_deliberately_failing():\n    assert False\n")
    reports_dir = tmp_path / "reports_e2e"

    def _generate_with_failing_gate(*, processed_dir):
        return phase7_cli._generate(processed_dir=processed_dir, test_files=(str(failing_test),))

    monkeypatch.setattr(phase7_cli, "_generate", _generate_with_failing_gate)

    exit_code = phase7_cli.run(["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)])
    assert exit_code == 1
    for filename in PROCESSED_OUTPUT_FILES:
        assert not (isolated_processed_dir / filename).exists()
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []


def test_a_failing_structural_step_writes_nothing_on_a_fresh_directory(isolated_processed_dir, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"

    def _always_fail(*, announcement_frame, pre_auction_frame, entries, protocol7):
        raise ValueError("deliberately induced structural failure")

    import treasury_auction_stress.evaluation.run_evaluation_phase7 as run_evaluation_phase7_module

    monkeypatch.setattr(run_evaluation_phase7_module, "run_full_phase7_evaluation", _always_fail)
    monkeypatch.setattr(phase7_cli, "run_full_phase7_evaluation", _always_fail)

    exit_code = phase7_cli.run(["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir)])
    assert exit_code == 1
    for filename in PROCESSED_OUTPUT_FILES:
        assert not (isolated_processed_dir / filename).exists()
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []


def test_atomic_write_parquet_does_not_touch_existing_file_on_failure(tmp_path, monkeypatch):
    final_path = tmp_path / "table.parquet"
    good_df = pd.DataFrame({"a": [1, 2, 3]})
    phase7_cli._atomic_write_parquet(good_df, final_path)
    original_bytes = final_path.read_bytes()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    bad_df = pd.DataFrame({"a": [999]})
    with pytest.raises(RuntimeError):
        phase7_cli._atomic_write_parquet(bad_df, final_path)

    assert final_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_text_does_not_touch_existing_file_on_failure(tmp_path, monkeypatch):
    final_path = tmp_path / "report.md"
    phase7_cli._atomic_write_text("original content", final_path)
    original = final_path.read_text()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(RuntimeError):
        phase7_cli._atomic_write_text("new content", final_path)

    assert final_path.read_text() == original
    assert list(tmp_path.glob(".*.tmp")) == []
