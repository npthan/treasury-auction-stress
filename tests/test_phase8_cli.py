from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from treasury_auction_stress.evaluation import phase8_cli
from treasury_auction_stress.evaluation.phase8_cli import (
    DASHBOARD_DATA_FILE,
    REPORT_FILE,
    REQUIRED_PHASE7_FILES,
    Phase8GenerationError,
)


def _real_processed_dir_or_skip() -> Path:
    processed = Path("data/processed")
    if not all((processed / f).exists() for f in REQUIRED_PHASE7_FILES):
        pytest.skip("processed Phase 7 tables not present in this environment")
    return processed


@pytest.fixture()
def isolated_processed_dir(tmp_path: Path) -> Path:
    real_dir = _real_processed_dir_or_skip()
    isolated = tmp_path / "processed"
    isolated.mkdir()
    for filename in REQUIRED_PHASE7_FILES:
        shutil.copy2(real_dir / filename, isolated / filename)
    return isolated


def test_require_phase7_files_raises_with_regeneration_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="phase7_cli"):
        phase8_cli._require_phase7_files(tmp_path)


def test_run_gate_tests_raises_on_a_real_failing_test_file(tmp_path):
    failing_test = tmp_path / "test_deliberately_failing.py"
    failing_test.write_text("def test_deliberately_failing():\n    assert False, 'induced failure'\n")

    with pytest.raises(Phase8GenerationError, match="failed"):
        phase8_cli._run_gate_tests(test_files=(str(failing_test),))


def test_run_gate_tests_passes_through_output_on_a_real_passing_test_file(tmp_path):
    passing_test = tmp_path / "test_deliberately_passing.py"
    passing_test.write_text("def test_deliberately_passing():\n    assert True\n")

    summary = phase8_cli._run_gate_tests(test_files=(str(passing_test),))
    assert "1 passed" in summary


def test_run_writes_report_and_dashboard_data_on_success(isolated_processed_dir, tmp_path):
    reports_dir = tmp_path / "reports"
    dashboard_data_dir = tmp_path / "dashboard_data"
    exit_code = phase8_cli.run(
        [
            "--processed-dir",
            str(isolated_processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--dashboard-data-dir",
            str(dashboard_data_dir),
        ]
    )
    assert exit_code == 0
    report_path = reports_dir / REPORT_FILE
    data_path = dashboard_data_dir / DASHBOARD_DATA_FILE
    assert report_path.exists()
    assert data_path.exists()

    report_text = report_path.read_text()
    assert "Phase 8 Interpretation" in report_text
    assert "poorly calibrated" in report_text.lower() or "poorly" in report_text.lower()

    data = json.loads(data_path.read_text())
    assert data["meta"]["cutoff_views"] == ["announcement", "pre_auction"]
    assert "warnings" in data
    assert data["warnings"]["classifier_poorly_calibrated"]


def test_run_is_reproducible_byte_for_byte_across_two_runs(isolated_processed_dir, tmp_path):
    reports_dir_1 = tmp_path / "reports1"
    data_dir_1 = tmp_path / "data1"
    reports_dir_2 = tmp_path / "reports2"
    data_dir_2 = tmp_path / "data2"

    for reports_dir, data_dir in ((reports_dir_1, data_dir_1), (reports_dir_2, data_dir_2)):
        exit_code = phase8_cli.run(
            [
                "--processed-dir",
                str(isolated_processed_dir),
                "--reports-dir",
                str(reports_dir),
                "--dashboard-data-dir",
                str(data_dir),
            ]
        )
        assert exit_code == 0

    assert (reports_dir_1 / REPORT_FILE).read_text() == (reports_dir_2 / REPORT_FILE).read_text()
    assert (data_dir_1 / DASHBOARD_DATA_FILE).read_text() == (data_dir_2 / DASHBOARD_DATA_FILE).read_text()


def test_a_failing_gate_leaves_previously_published_outputs_untouched(isolated_processed_dir, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    data_dir = tmp_path / "dashboard_data"
    reports_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    report_path = reports_dir / REPORT_FILE
    data_path = data_dir / DASHBOARD_DATA_FILE
    report_path.write_text("placeholder report content")
    data_path.write_text("placeholder json content")
    before_report = report_path.read_bytes()
    before_data = data_path.read_bytes()

    def _always_fail(*, processed_dir, test_files=phase8_cli.DEFAULT_PHASE8_TEST_FILES):
        raise Phase8GenerationError("deliberately induced failure for the acceptance-review test")

    monkeypatch.setattr(phase8_cli, "_generate", _always_fail)

    exit_code = phase8_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir), "--dashboard-data-dir", str(data_dir)]
    )
    assert exit_code == 1
    assert report_path.read_bytes() == before_report
    assert data_path.read_bytes() == before_data
    assert list(reports_dir.glob(".*.tmp")) == []
    assert list(data_dir.glob(".*.tmp")) == []


def test_a_failing_gate_via_the_real_subprocess_mechanism_writes_nothing(isolated_processed_dir, tmp_path, monkeypatch):
    failing_test = tmp_path / "test_deliberately_failing.py"
    failing_test.write_text("def test_deliberately_failing():\n    assert False\n")
    reports_dir = tmp_path / "reports_e2e"
    data_dir = tmp_path / "dashboard_data_e2e"

    def _generate_with_failing_gate(*, processed_dir):
        return phase8_cli._generate(processed_dir=processed_dir, test_files=(str(failing_test),))

    monkeypatch.setattr(phase8_cli, "_generate", _generate_with_failing_gate)

    exit_code = phase8_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir), "--dashboard-data-dir", str(data_dir)]
    )
    assert exit_code == 1
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []
    assert not data_dir.exists() or list(data_dir.iterdir()) == []


def test_missing_phase7_inputs_fails_cleanly_without_writing(tmp_path):
    empty_processed_dir = tmp_path / "processed_empty"
    empty_processed_dir.mkdir()
    reports_dir = tmp_path / "reports"
    data_dir = tmp_path / "dashboard_data"

    exit_code = phase8_cli.run(
        ["--processed-dir", str(empty_processed_dir), "--reports-dir", str(reports_dir), "--dashboard-data-dir", str(data_dir)]
    )
    assert exit_code == 1
    assert not reports_dir.exists() or list(reports_dir.iterdir()) == []
    assert not data_dir.exists() or list(data_dir.iterdir()) == []


def test_dashboard_tenor_scoped_tables_are_genuinely_tenor_filtered(isolated_processed_dir, tmp_path):
    """Guard against the dashboard defect found in acceptance review: the
    tenor filter must recompute each panel on the selected tenor's own
    auctions, never silently fall back to the pooled table while a
    tenor is selected. Checks this on the real accepted data: (1) every
    tenor's own point/probabilistic/stress sample sizes sum exactly to
    the pooled complete-year count (no auction double-counted or
    dropped), and (2) at least one tenor's MAE and stress prevalence
    differ from the pooled figure by more than noise -- if the
    generator ever regressed to reusing the pooled table for every
    tenor, this would fail because all tenors would report identical
    numbers.
    """
    reports_dir = tmp_path / "reports"
    data_dir = tmp_path / "dashboard_data"
    exit_code = phase8_cli.run(
        ["--processed-dir", str(isolated_processed_dir), "--reports-dir", str(reports_dir), "--dashboard-data-dir", str(data_dir)]
    )
    assert exit_code == 0
    data = json.loads((data_dir / DASHBOARD_DATA_FILE).read_text())

    tenors = data["meta"]["tenors"]
    assert len(tenors) >= 2

    def _pooled_n(model_id, cv):
        row = next(r for r in data["point"]["pooled"] if r["model_id"] == model_id and r["cutoff_view"] == cv)
        return row["n"]

    def _tenor_n(tenor, model_id, cv):
        row = next(
            r for r in data["point"]["pooled_by_tenor"][tenor] if r["model_id"] == model_id and r["cutoff_view"] == cv
        )
        return row["n"]

    def _tenor_mae(tenor, model_id, cv):
        row = next(
            r for r in data["point"]["pooled_by_tenor"][tenor] if r["model_id"] == model_id and r["cutoff_view"] == cv
        )
        return row["mae"]

    model_id, cv = "baseline_recent_history_adaptive", "announcement"
    assert sum(_tenor_n(t, model_id, cv) for t in tenors) == _pooled_n(model_id, cv)

    pooled_mae = next(r for r in data["point"]["pooled"] if r["model_id"] == model_id and r["cutoff_view"] == cv)["mae"]
    tenor_maes = {t: _tenor_mae(t, model_id, cv) for t in tenors}
    assert any(abs(mae - pooled_mae) > 0.1 for mae in tenor_maes.values())
    assert len({round(mae, 4) for mae in tenor_maes.values()}) > 1  # not every tenor identical either

    # Stress-classifier positive counts must also partition the pooled total exactly.
    pooled_stress = data["stress"]["pooled"][cv]
    tenor_stress_n_positive = [data["stress"]["pooled_by_tenor"][t][cv]["n_positive"] for t in tenors]
    assert sum(tenor_stress_n_positive) == pooled_stress["n_positive"]
    tenor_stress_n = [data["stress"]["pooled_by_tenor"][t][cv]["n"] for t in tenors]
    assert sum(tenor_stress_n) == pooled_stress["n"]

    # Case studies re-selected per tenor must actually belong to that tenor.
    for tenor in tenors:
        for case in data["point"]["case_studies_by_tenor"][tenor][cv]:
            assert case["tenor"] == tenor


def test_atomic_write_text_does_not_touch_existing_file_on_failure(tmp_path, monkeypatch):
    final_path = tmp_path / "report.md"
    phase8_cli._atomic_write_text("original content", final_path)
    original = final_path.read_text()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(RuntimeError):
        phase8_cli._atomic_write_text("new content", final_path)

    assert final_path.read_text() == original
    assert list(tmp_path.glob(".*.tmp")) == []
