"""Regression tests for configs/release_manifest.yml: the tracked tree
must match the declared public-release manifest exactly, directories
that must stay untracked (reports/, artifacts/, real data/) must have
no tracked file, and the committed dashboard aggregate must match its
recorded SHA-256. Tests that need `git ls-files` skip gracefully in a
git-free export, matching the other governance tests' convention.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "release_manifest.yml"


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (git-free export); tracked-file checks need git")
    return [line for line in out.stdout.splitlines() if line]


def test_tracked_file_counts_match_manifest_exactly():
    manifest = _manifest()["tracked_file_counts"]
    tracked = _tracked_files()
    by_location: dict[str, int] = {}
    for path in tracked:
        top = path.split("/", 1)[0] if "/" in path else "top_level"
        by_location[top] = by_location.get(top, 0) + 1
    expected = {k: v for k, v in manifest.items() if k != "total"}
    assert by_location == expected, (
        "tracked tree no longer matches configs/release_manifest.yml -- "
        "update the manifest deliberately if this change is intended"
    )
    assert len(tracked) == manifest["total"]


def test_no_tracked_file_under_a_must_stay_untracked_prefix():
    prefixes = _manifest()["must_stay_untracked_prefixes"]
    tracked = _tracked_files()
    offenders = [
        path
        for path in tracked
        for prefix in prefixes
        if path.startswith(prefix) and not path.endswith(".gitkeep")
    ]
    assert offenders == [], f"tracked files under a must-stay-untracked prefix: {offenders}"


def test_no_tracked_reports_directory_remains():
    tracked = _tracked_files()
    assert not any(p == "reports" or p.startswith("reports/") for p in tracked)


def test_artifacts_directory_is_gitignored_and_not_tracked():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^artifacts/$", gitignore, re.MULTILINE), (
        ".gitignore must ignore the generated-output artifacts/ directory"
    )
    tracked = _tracked_files()
    assert not any(p.startswith("artifacts/") for p in tracked)


def test_committed_dashboard_aggregate_matches_recorded_sha256():
    recorded = _manifest()["committed_artifact_checksums"]
    for rel_path, digest in recorded.items():
        content = (REPO_ROOT / rel_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest, (
            f"{rel_path} no longer matches its recorded SHA-256 -- if this change "
            "is deliberate, update configs/release_manifest.yml in the same commit"
        )


def test_accepted_artifact_checksums_are_well_formed():
    accepted = _manifest()["accepted_local_artifact_checksums"]
    assert accepted, "accepted-artifact checksum section must not be empty"
    for group, entries in accepted.items():
        for name, digest in entries.items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), (group, name, digest)


def test_readme_repo_path_references_all_resolve():
    """Every repository-relative path the README points at must exist --
    a broken-link guard now that generated reports are untracked."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(
        re.findall(
            r"`((?:docs|configs|dashboard|tests|src|\.github)/[A-Za-z0-9_/.\-]+)`",
            readme,
        )
    )
    assert referenced, "expected the README to reference repository paths"
    missing = sorted(p for p in referenced if not (REPO_ROOT / p).exists())
    assert missing == [], f"README references nonexistent paths: {missing}"
    assert "reports/" not in readme, "README must not reference the removed reports/ directory"
