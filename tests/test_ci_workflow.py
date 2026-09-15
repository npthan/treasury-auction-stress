"""Static configuration checks for .github/workflows/ci.yml.

These do not run the workflow (no GitHub Actions runner in this test
suite) -- they parse the committed YAML and assert the safety properties
Phase 9 requires: no secrets, least-privilege permissions, Python 3.12,
locked dependency install, offline tests only, pinned official actions.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CI_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_ci_workflow_file_exists_and_parses():
    assert CI_WORKFLOW_PATH.exists()
    workflow = _load_workflow()
    assert isinstance(workflow, dict)


def test_ci_workflow_has_read_only_permissions():
    workflow = _load_workflow()
    assert workflow.get("permissions") == {"contents": "read"}


def test_ci_workflow_triggers_are_push_and_pull_request_only():
    workflow = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True.
    triggers = workflow.get(True, workflow.get("on"))
    assert set(triggers) == {"push", "pull_request"}


def test_ci_workflow_uses_python_3_12():
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in raw


def test_ci_workflow_installs_locked_dependencies():
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "uv sync --locked" in raw


def test_ci_workflow_runs_ruff_and_default_pytest():
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "uv run ruff check" in raw
    assert "uv run pytest" in raw


def test_ci_workflow_never_invokes_live_network_marker():
    """The CI job may *mention* live_network in a comment (documenting why
    it's excluded) but must never run a step that actually selects it."""
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    run_commands = [
        step["run"]
        for job in jobs.values()
        for step in job.get("steps", [])
        if "run" in step
    ]
    assert run_commands, "expected at least one run step"
    for command in run_commands:
        assert "live_network" not in command
        assert 'addopts=""' not in command


def test_ci_workflow_never_references_repository_secrets():
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in raw
    assert "FRED_API_KEY" not in raw


_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_ci_workflow_actions_are_pinned_to_an_immutable_commit_sha():
    """Acceptance-review remediation (finding 7): a version TAG like
    `@v7.0.1` is not floating in the sense of silently tracking new
    releases, but it IS mutable -- the upstream repo owner can force-move
    the tag to point at different, unreviewed code, and this workflow
    would silently pull it. Only a full 40-hex-character commit SHA is
    immutable. A merely non-floating but still-mutable tag (e.g. `v7`,
    `v7.0.1`, or even a per-major-version tag) must NOT satisfy this
    check -- only a real SHA does.
    """
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    steps = [step for job in jobs.values() for step in job.get("steps", []) if "uses" in step]
    assert steps, "expected at least one action step"
    for step in steps:
        ref = step["uses"]
        name, _, version = ref.partition("@")
        assert version, f"action {name!r} is not pinned to a ref"
        assert version not in {"main", "master", "latest"}, (
            f"action {name!r} is pinned to a floating/unsafe ref {version!r}"
        )
        assert _FULL_COMMIT_SHA.match(version), (
            f"action {name!r} is pinned to {version!r}, which is not a full "
            "40-character commit SHA -- a version tag (even a specific, "
            "non-floating one) is mutable and does not satisfy the "
            "immutable-pin requirement"
        )


def test_ci_workflow_mutable_tag_pin_is_rejected_by_the_immutability_check(tmp_path):
    """Direct proof that a merely non-floating tag cannot slip past the
    immutability check above: construct a synthetic workflow pinned to a
    specific version TAG (not a SHA) and assert the same assertion this
    module's real test applies would fail for it."""
    synthetic = tmp_path / "ci.yml"
    synthetic.write_text(
        "name: CI\n"
        "on: [push]\n"
        "permissions: {contents: read}\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7.0.1\n"
    )
    workflow = yaml.safe_load(synthetic.read_text())
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    ]
    ref = steps[0]["uses"]
    _name, _, version = ref.partition("@")
    assert not _FULL_COMMIT_SHA.match(version), (
        "a version tag must never pass the immutable-SHA check"
    )


def test_ci_workflow_never_downloads_bulk_source_data():
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "treasury_auction_stress.data.cli",
        "fiscaldata.treasury.gov",
        "markets.newyorkfed.org",
        "home.treasury.gov",
        "publicreporting.cftc.gov",
        "philadelphiafed.org",
    ):
        assert forbidden not in raw
