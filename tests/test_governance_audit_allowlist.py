"""Regression tests for configs/governance_audit_allowlist.yml and its
loader (`audit_cli._load_allowlist_fingerprints`).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from treasury_auction_stress.governance.audit_cli import _load_allowlist_fingerprints
from treasury_auction_stress.governance.history_audit import audit_history
from treasury_auction_stress.governance.release_audit import (
    allowlist_token,
    audit_tree,
    finding_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "configs" / "governance_audit_allowlist.yml"


def test_allowlist_file_parses_and_every_entry_has_a_fingerprint_and_reason():
    data = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    entries = data["allowed_fingerprints"]
    assert len(entries) > 0
    for entry in entries:
        assert entry["fingerprint"].startswith("fp:")
        assert entry.get("reason")
        assert entry.get("file")
        # Path scoping is mandatory: every entry names the exact
        # repo-relative path(s) its value is approved at.
        paths = entry.get("paths")
        assert paths and isinstance(paths, list)
        for path in paths:
            assert isinstance(path, str) and path and not path.startswith("/")
            assert "::" not in path


def test_allowlist_entries_are_unique():
    data = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    fingerprints = [e["fingerprint"] for e in data["allowed_fingerprints"]]
    assert len(fingerprints) == len(set(fingerprints))


def test_loader_returns_every_path_scoped_token_in_the_file():
    data = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    expected = {
        f"{path}::{e['fingerprint']}"
        for e in data["allowed_fingerprints"]
        for path in e["paths"]
    }
    assert _load_allowlist_fingerprints(REPO_ROOT) == expected


def test_loader_returns_empty_set_when_no_config_file_present(tmp_path):
    assert _load_allowlist_fingerprints(tmp_path) == frozenset()


def test_allowlist_file_itself_does_not_reproduce_its_own_trigger_patterns():
    """The allowlist's own prose must describe the false positives it
    documents without reproducing the exact trigger-shaped substrings --
    otherwise the file would perpetually re-trigger the very findings it
    exists to explain. Regression guard for exactly that defect, found
    and fixed while authoring this file."""
    text = ALLOWLIST_PATH.read_text(encoding="utf-8")
    result = audit_tree([("configs/governance_audit_allowlist.yml", text)])
    hard_blocks = [f for f in result.findings if f.severity == "hard_block"]
    assert hard_blocks == [], (
        "the allowlist file's own prose re-triggers a hard-blocking finding: "
        f"{[(f.category, f.location) for f in hard_blocks]}"
    )


def test_public_release_current_tree_audit_passes_with_the_allowlist_applied():
    """End-to-end: the actual proposed public tree, scanned with the
    reviewed allowlist applied exactly as audit_cli.py applies it, must
    pass with zero hard-blocking findings."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    paths_and_contents = []
    for rel_path in sorted(set(tracked) | set(untracked)):
        full_path = REPO_ROOT / rel_path
        if not full_path.is_file():
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = ""
        paths_and_contents.append((rel_path, text))

    allowed = _load_allowlist_fingerprints(REPO_ROOT)
    result = audit_tree(paths_and_contents, allowed_identifying_info=allowed)
    unallowlisted = [
        f
        for f in result.findings
        if f.severity == "hard_block"
        and allowlist_token(finding_path(f), str(f.fingerprint)) not in allowed
    ]
    assert unallowlisted == []
    assert result.passed is True


def test_history_audit_still_blocks_by_default_with_no_allowlist_passed():
    """audit_history()'s own default (no allowlist argument) surfaces
    every hard-blocking finding unfiltered -- an allowlist only ever
    suppresses a finding when a caller (audit_cli.py, using the same
    reviewed fingerprints as the current-tree scan) explicitly passes
    one. This test proves the function's default is the strict,
    nothing-suppressed behavior, independent of whatever audit_tree
    elsewhere is doing."""
    result = audit_history(REPO_ROOT)
    hard_blocks = {f.category for f in result.findings if f.severity == "hard_block"}
    assert "identifying_info:email_address" in hard_blocks
    assert "identifying_info:absolute_home_path" in hard_blocks
    assert result.passed is False
