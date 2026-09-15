"""CLI entry point for the Phase 9 release-governance audit.

Scans (1) the current proposed public tree -- every git-tracked path
still present on disk, plus every new, not-yet-staged, not-gitignored
path (i.e. what `git add -A` would actually stage right now, not the
possibly-stale index alone) -- and (2) the full reachable git history
(every local branch/tag, every distinct blob) for secrets, credentials,
and non-consensual identifying information, and prints a redacted,
category-only report. Never prints an actual matched value. Exits 0
only if BOTH scans are free of hard-blocking findings.

The explicit, per-value, per-path fingerprint allowlist
(`configs/governance_audit_allowlist.yml` and/or `--allow-fingerprint`,
both using '<repo-relative-path>::fp:<hex>' tokens) applies to BOTH
scans identically -- a specific reviewed value at its reviewed path
(this project's own deliberately-fake test canaries, or documented
benign prose) is not suddenly sensitive just because it also happens
to be reachable in history, while the SAME value at any other path
still blocks in both scans. A `rejected_path` or `oversized_fixture` finding (no single
value to review, hence no fingerprint) can never be allowlisted in
either scan.

Usage:
    uv run python -m treasury_auction_stress.governance.audit_cli
    uv run python -m treasury_auction_stress.governance.audit_cli \\
        --allow-fingerprint tests/test_example.py::fp:0123456789ab
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from .history_audit import HistoryFinding, audit_history
from .release_audit import (
    Finding,
    audit_tree,
    find_large_tracked_files,
    is_publishable_fixture_size,
)


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _load_allowlist_fingerprints(repo_root: Path) -> frozenset[str]:
    """Load the explicit, human-reviewed allowlist from
    `configs/governance_audit_allowlist.yml`, if present, as path-scoped
    tokens ('<repo-relative-path>::fp:<hex>'). Every entry is a single
    reviewer's individual, justified decision that one specific matched
    value AT ONE OR MORE specific, listed paths (never a bare category,
    bare directory, or value-anywhere exemption) is not sensitive -- see
    that file's own header for the review discipline. Returns an empty
    set if the file doesn't exist (nothing is allowed by default).
    """
    config_path = repo_root / "configs" / "governance_audit_allowlist.yml"
    if not config_path.exists():
        return frozenset()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = data.get("allowed_fingerprints", [])
    return frozenset(
        f"{path}::{e['fingerprint']}"
        for e in entries
        if "fingerprint" in e
        for path in e.get("paths", [])
    )


def _tracked_paths_and_contents(repo_root: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Every path that would be tracked after `git add -A` -- the real
    current working-tree state, not the stale index. Plain `git
    ls-files` alone still lists a path that was deleted on disk but not
    yet staged (e.g. an intentionally-removed fixture) as if it still
    existed, which would hide exactly the class of "is anything unsafe
    on disk right now" question this audit exists to answer; `git
    ls-files --others --exclude-standard` adds new, not-yet-staged,
    not-gitignored files the same way.

    Returned content is text where decodable, or "" for binary/
    undecodable files. Binary files still get "" (not skipped entirely)
    so `audit_tree`'s path-allowlist check -- which a binary secret/
    credential/archive file must not bypass -- still runs on them; only
    their *content* can't be regex-scanned as text.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    all_paths = sorted(set(tracked) | set(untracked))
    paths_and_contents: list[tuple[str, str]] = []
    existing_paths: list[str] = []
    for rel_path in all_paths:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue  # deleted-but-still-indexed; not actually on disk
        existing_paths.append(rel_path)
        try:
            text = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = ""
        paths_and_contents.append((rel_path, text))
    return paths_and_contents, existing_paths


def _print_findings(findings: list[Finding] | list[HistoryFinding]) -> None:
    if not findings:
        print("    none")
        return
    for f in findings:
        location = f.location if isinstance(f, Finding) else f"{f.example_commit}:{f.path}"
        print(f"    [{f.severity}] {f.category} @ {location} -- {f.description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-fingerprint",
        action="append",
        default=[],
        metavar="PATH::fp:HEX",
        help=(
            "Explicitly allow one specific finding (secret OR "
            "identifying-info) by its path-scoped token: the finding's "
            "repo-relative path, '::', then its fp:... fingerprint (both "
            "from a prior run's output). IN ADDITION to configs/"
            "governance_audit_allowlist.yml if present. Never accepts a "
            "bare category, bare path, or bare value-anywhere fingerprint "
            "-- each token is a single reviewed, explicit decision for "
            "one value at one path; the same value anywhere else still "
            "blocks. "
            "Repeatable. Applies to BOTH the current-tree and full-history "
            "audits below -- a `rejected_path` or `oversized_fixture` "
            "finding can never be suppressed this way in either audit "
            "(they have no single value to review), but a specific "
            "secret/identifying-info fingerprint, once reviewed, is not "
            "suppressed in one scan and re-flagged in the other."
        ),
    )
    args = parser.parse_args(argv)
    for token in args.allow_fingerprint:
        if "::fp:" not in token:
            parser.error(
                f"--allow-fingerprint {token!r} is not path-scoped; "
                "expected '<repo-relative-path>::fp:<hex>'"
            )
    repo_root = _repo_root()
    allowed = frozenset(args.allow_fingerprint) | _load_allowlist_fingerprints(repo_root)

    print("=== Current-tree audit ===")
    paths_and_contents, all_paths = _tracked_paths_and_contents(repo_root)
    tree_result = audit_tree(paths_and_contents, allowed_identifying_info=allowed)
    print(f"  files scanned: {len(paths_and_contents)}")
    print("  findings:")
    _print_findings(tree_result.findings)
    oversized = find_large_tracked_files(all_paths, repo_root)
    if oversized:
        print("  large tracked files (>=1MB):")
        for path, size in oversized:
            print(f"    {path}: {size} bytes")
    oversized_fixtures = [
        (path, size)
        for path, size in ((p, (repo_root / p).stat().st_size) for p in all_paths)
        if not is_publishable_fixture_size(path, size)
    ]
    if oversized_fixtures:
        print("  oversized fixtures (over the tests/fixtures/ size cap):")
        for path, size in oversized_fixtures:
            print(f"    {path}: {size} bytes")
    tree_passed = tree_result.passed and not oversized_fixtures
    print(f"  verdict: {'PASSED' if tree_passed else 'BLOCKED'}")

    print()
    print("=== Full-history audit (all reachable refs) ===")
    # The same explicit, per-value allowlist is applied here as above --
    # symmetric with audit_tree. A finding with no fingerprint
    # (rejected_path, oversized_fixture) is still never suppressible;
    # only a specific, individually-reviewed secret/identifying-info
    # value can be, in either scan.
    history_result = audit_history(repo_root, allowed_identifying_info=allowed)
    print(f"  unique blobs scanned: {history_result.blobs_scanned}")
    print("  findings:")
    _print_findings(history_result.findings)
    print(f"  verdict: {'PASSED' if history_result.passed else 'BLOCKED'}")

    overall_passed = tree_passed and history_result.passed
    print()
    print(f"OVERALL: {'PASSED' if overall_passed else 'BLOCKED'}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
