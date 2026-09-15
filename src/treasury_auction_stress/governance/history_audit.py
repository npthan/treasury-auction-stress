"""Full reachable-git-history secret/PII/bulk-data audit.

Phase 9 acceptance-review remediation: the original Phase 9 tree/history
review (the Phase 9 release audit Section 3) scanned the
*current* tree, `git log --all --diff-filter=D` (files added then later
*deleted*), and a largest-objects-by-size inventory -- but never every
reachable blob's own content. That misses the exact defect class found
in this repository: a file that was committed with sensitive content,
then *modified in place* (not deleted) to redact it. The earlier,
now-redacted blob is still a distinct, fully reachable git object; only
a full blob walk finds it.

This module walks every commit reachable from the given refs (default:
`--all`, i.e. every local branch and tag, not just the current branch),
lists each commit's *entire* tree (not just its diff, so an old version
of a file that still exists today is still visited), and scans each
distinct blob's content exactly once (deduplicated by blob sha) using
the same secret/identifying-info/path-policy checks `release_audit.py`
applies to the current tree. Only read-only git plumbing is used
(`rev-list`, `ls-tree`, `cat-file`) -- no `checkout`, `switch`, `reset`,
`gc`, or `prune` is ever run, and no historical commit is checked out.

As with `release_audit.py`, no function here ever embeds an actual
matched secret/PII value in a returned finding -- only its category,
an example commit and repo-relative path, a line number when
available, and a non-reversible fingerprint.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .release_audit import (
    Finding,
    allowlist_token,
    is_publishable_fixture_size,
    is_publishable_path,
    scan_for_identifying_info,
    scan_for_secret_patterns,
)

_LARGE_OBJECT_REVIEW_THRESHOLD_BYTES = 1_000_000


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout


@dataclass(frozen=True)
class BlobRecord:
    """One unique blob (deduplicated by content sha) and every distinct
    repo-relative path / example commit it was ever recorded at across
    the walked history. `commits` is capped (a handful of examples, not
    every commit that ever carried this exact content) -- reporting
    needs one resolvable commit/path pair, not an exhaustive list.
    """

    sha: str
    size: int
    paths: tuple[str, ...]
    commits: tuple[str, ...]


def enumerate_reachable_blobs(
    repo_root: Path, refs: tuple[str, ...] = ("--all",)
) -> dict[str, BlobRecord]:
    """Enumerate every distinct blob reachable from `refs` (default:
    every local ref -- branches and tags). For each commit reachable
    from those refs, lists that commit's *complete* tree (`git ls-tree
    -r -l`), not just files it changed, so a blob that was later
    modified (its path's current content differs) but whose OLD content
    is still reachable in an earlier commit's tree is still found.
    Read-only: `rev-list` + `ls-tree` only, never `checkout`/`gc`/`prune`.
    """
    commits = [c for c in _run_git(["rev-list", *refs], cwd=repo_root).split("\n") if c]
    blob_paths: dict[str, set[str]] = {}
    blob_commits: dict[str, list[str]] = {}
    blob_sizes: dict[str, int] = {}

    for commit in commits:
        out = _run_git(["ls-tree", "-r", "-l", commit], cwd=repo_root)
        for line in out.splitlines():
            if not line:
                continue
            meta, _, path = line.partition("\t")
            fields = meta.split()
            if len(fields) < 4:
                continue
            _mode, obj_type, obj_sha, obj_size = fields[:4]
            if obj_type != "blob":
                continue
            blob_paths.setdefault(obj_sha, set()).add(path)
            example_commits = blob_commits.setdefault(obj_sha, [])
            if commit not in example_commits and len(example_commits) < 3:
                example_commits.append(commit)
            if obj_sha not in blob_sizes and obj_size.isdigit():
                blob_sizes[obj_sha] = int(obj_size)

    return {
        sha: BlobRecord(
            sha=sha,
            size=blob_sizes.get(sha, -1),
            paths=tuple(sorted(paths)),
            commits=tuple(blob_commits.get(sha, ())),
        )
        for sha, paths in blob_paths.items()
    }


def _read_blob_text(repo_root: Path, sha: str) -> str | None:
    """Blob content as UTF-8 text, or None if not decodable (binary --
    still covered by the path/size checks in `audit_history`, just not
    scanned line-by-line for text patterns)."""
    result = subprocess.run(
        ["git", "cat-file", "-p", sha], cwd=repo_root, capture_output=True, check=True
    )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass(frozen=True)
class HistoryFinding:
    """One full-history finding. Never embeds the actual matched value
    -- see `Finding` in `release_audit.py`, which this mirrors."""

    category: str
    severity: str
    example_commit: str
    path: str
    line: int | None
    fingerprint: str | None
    description: str


@dataclass(frozen=True)
class HistoryAuditResult:
    passed: bool
    blobs_scanned: int
    findings: list[HistoryFinding] = field(default_factory=list)


def _to_history_finding(f: Finding, example_commit: str) -> HistoryFinding:
    path, _, lineno_str = f.location.rpartition(":")
    return HistoryFinding(
        category=f.category,
        severity=f.severity,
        example_commit=example_commit,
        path=path or f.location,
        line=int(lineno_str) if lineno_str.isdigit() else None,
        fingerprint=f.fingerprint,
        description=f.description,
    )


def audit_history(
    repo_root: Path,
    refs: tuple[str, ...] = ("--all",),
    allowed_identifying_info: frozenset[str] = frozenset(),
) -> HistoryAuditResult:
    """Scan every distinct blob reachable from `refs` for secrets,
    hard-blocking identifying information, forbidden/rejected paths, and
    oversized fixtures/objects. `passed=False` if any hard-blocking
    finding exists whose path-scoped allowlist token
    (`release_audit.allowlist_token`: '<path>::<fp:...>') is not
    explicitly listed in `allowed_identifying_info` -- the same explicit,
    per-value, per-path,
    never-inferred-from-presence policy `release_audit.audit_tree` uses,
    and applied symmetrically here to BOTH `secret:*` and
    `identifying_info:*` categories (this used to check only
    identifying-info categories, leaving `secret:*` matches permanently
    unallowlistable in history even when the exact same fingerprint was
    already a reviewed, documented false positive for the current-tree
    audit -- see `configs/governance_audit_allowlist.yml`; fixed so a
    project's own deliberately-fake test canaries can be committed and
    still pass full-history audit). A `rejected_path` or
    `oversized_fixture` finding can never be allowlisted this way (their
    `fingerprint` is always `None`).
    """
    blobs = enumerate_reachable_blobs(repo_root, refs=refs)
    findings: list[HistoryFinding] = []

    for sha, record in blobs.items():
        example_commit = record.commits[0] if record.commits else "<unresolved>"

        for path in record.paths:
            if not is_publishable_path(path):
                findings.append(
                    HistoryFinding(
                        category="rejected_path",
                        severity="hard_block",
                        example_commit=example_commit,
                        path=path,
                        line=None,
                        fingerprint=None,
                        description=(
                            f"historical blob at '{path}' is not on the "
                            "public-release allowlist (may have since been "
                            "modified/redacted at this path, but the object "
                            "itself remains reachable in history)"
                        ),
                    )
                )
            size = max(record.size, 0)
            if not is_publishable_fixture_size(path, size):
                findings.append(
                    HistoryFinding(
                        category="oversized_fixture",
                        severity="hard_block",
                        example_commit=example_commit,
                        path=path,
                        line=None,
                        fingerprint=None,
                        description=(
                            f"historical blob at '{path}' is {size} bytes, "
                            "over the tests/fixtures/ size cap"
                        ),
                    )
                )

        if record.size > _LARGE_OBJECT_REVIEW_THRESHOLD_BYTES:
            findings.append(
                HistoryFinding(
                    category="large_object",
                    severity="review",
                    example_commit=example_commit,
                    path=record.paths[0] if record.paths else "<unknown>",
                    line=None,
                    fingerprint=None,
                    description=f"historical blob is {record.size} bytes",
                )
            )

        text = _read_blob_text(repo_root, sha)
        if text is None:
            continue

        for path in record.paths:
            for f in scan_for_secret_patterns(text, location=path):
                if (
                    f.fingerprint is not None
                    and allowlist_token(path, f.fingerprint) in allowed_identifying_info
                ):
                    continue
                findings.append(_to_history_finding(f, example_commit))
            for f in scan_for_identifying_info(text, location=path):
                if (
                    f.fingerprint is not None
                    and allowlist_token(path, f.fingerprint) in allowed_identifying_info
                ):
                    continue
                findings.append(_to_history_finding(f, example_commit))

    passed = not any(f.severity == "hard_block" for f in findings)
    return HistoryAuditResult(passed=passed, blobs_scanned=len(blobs), findings=findings)
