"""Regression tests for the full-reachable-history governance audit
(`treasury_auction_stress.governance.history_audit`).

These build small, throwaway, synthetic git repositories under pytest's
`tmp_path` -- never anything in this project's own history -- and prove
the specific defect class the Phase 9 acceptance review found: a secret
or local home path that was committed and later modified/redacted in
place (never deleted) is still detected, because the object graph still
contains the earlier blob.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from treasury_auction_stress.governance.history_audit import (
    audit_history,
    enumerate_reachable_blobs,
)

CANARY_SECRET = "AKIACANARY0123456789"  # AWS-access-key-shaped, unmistakably fake
CANARY_HOME_PATH = "/Users/definitely_not_a_real_username_9f8e7d/Projects/x"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _commit(repo: Path, rel_path: str, content: str, message: str) -> str:
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    _git(repo, "add", rel_path)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _init_repo(tmp_path)


def test_modified_but_not_deleted_secret_is_still_found_in_history(repo: Path):
    """The core Phase 9 acceptance-review defect: a secret is committed,
    then the file is edited to remove it (never deleted) in a later
    commit. A current-tree-only scan sees nothing; the history audit
    must still find the earlier blob."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")

    result = audit_history(repo)

    assert result.passed is False
    secret_findings = [f for f in result.findings if f.category == "secret:aws_access_key"]
    assert len(secret_findings) == 1
    assert secret_findings[0].path == "config.py"
    assert CANARY_SECRET not in secret_findings[0].description


def test_modified_but_not_deleted_home_path_is_still_found_in_history(repo: Path):
    """Same defect class, but for the actual category this project's
    review reproduced: an absolute /Users/... path redacted in a later
    commit but never removed from history."""
    _commit(
        repo,
        "report.md",
        f"ran from {CANARY_HOME_PATH}/.venv/bin/python3\n",
        "record local run",
    )
    _commit(repo, "report.md", "ran from <repo-root>/.venv/bin/python3\n", "redact path")

    result = audit_history(repo)

    assert result.passed is False
    home_path_findings = [
        f for f in result.findings if f.category == "identifying_info:absolute_home_path"
    ]
    assert len(home_path_findings) == 1
    assert CANARY_HOME_PATH not in home_path_findings[0].description
    assert home_path_findings[0].fingerprint is not None


def test_current_tree_alone_would_have_missed_the_redacted_secret(repo: Path):
    """Contrast case proving the defect is real: scanning ONLY the final
    commit's tree (what the original Phase 9 review effectively did)
    finds nothing, while the history audit does."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")

    current_tree_text = (repo / "config.py").read_text()
    assert CANARY_SECRET not in current_tree_text  # the working tree is clean

    result = audit_history(repo)
    assert result.passed is False  # but history still carries it


def test_deduplicates_identical_blobs_across_commits(repo: Path):
    """The same secret-bearing content committed unchanged across three
    commits (e.g. an unrelated file also touched each time) must be
    scanned once, not three times -- blob content is deduplicated by
    sha, not walked once per commit."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "other.txt", "unrelated change 1\n", "unrelated 1")
    _commit(repo, "other.txt", "unrelated change 2\n", "unrelated 2")

    result = audit_history(repo)

    secret_findings = [f for f in result.findings if f.category == "secret:aws_access_key"]
    assert len(secret_findings) == 1  # not 3


def test_reachable_from_a_branch_never_merged_to_main_is_still_found(repo: Path):
    """A secret committed only on a side branch (never merged, not an
    ancestor of main) must still be caught -- `--all` covers every local
    ref, not just the current branch."""
    _commit(repo, "README.md", "hello\n", "init")
    _git(repo, "checkout", "-q", "-b", "feature/leaky")
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "leak on a branch")
    _git(repo, "checkout", "-q", "main")

    main_only = audit_history(repo, refs=("main",))
    assert main_only.passed is True  # not reachable from main alone

    all_refs = audit_history(repo)  # default refs=("--all",)
    assert all_refs.passed is False
    assert any(f.category == "secret:aws_access_key" for f in all_refs.findings)


def test_reachable_only_from_a_tag_is_still_found(repo: Path):
    """A secret reachable only via a tag (e.g. an old release tag whose
    commit isn't on any branch anymore) must still be caught."""
    _commit(repo, "README.md", "hello\n", "init")
    sha = _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "leak, then tag it")
    _git(repo, "tag", "v0.0.1-leaky", sha)
    # Move main off that commit so it's reachable ONLY via the tag.
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact on main")

    result = audit_history(repo)  # default refs=("--all",) includes tags

    assert result.passed is False
    assert any(f.category == "secret:aws_access_key" for f in result.findings)


def test_history_audit_never_checks_out_a_historical_commit(repo: Path):
    """Read-only guarantee: HEAD, the current branch, and the working
    tree must be byte-identical before and after a full-history audit."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")

    before_head = _git(repo, "rev-parse", "HEAD").stdout
    before_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout
    before_status = _git(repo, "status", "--porcelain").stdout
    before_working_tree = (repo / "config.py").read_text()

    audit_history(repo)

    assert _git(repo, "rev-parse", "HEAD").stdout == before_head
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout == before_branch
    assert _git(repo, "status", "--porcelain").stdout == before_status
    assert (repo / "config.py").read_text() == before_working_tree


def test_history_audit_only_uses_read_only_git_plumbing(repo: Path, monkeypatch):
    """Spy on every subprocess git invocation made during a full-history
    audit and assert none of them is a mutating/history-rewriting/
    checkout-style command."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")

    forbidden = {
        "checkout",
        "switch",
        "reset",
        "clean",
        "gc",
        "prune",
        "commit",
        "push",
        "rebase",
        "filter-branch",
        "filter-repo",
    }
    invoked_subcommands: list[str] = []
    real_run = subprocess.run

    def spy_run(args, *a, **kw):
        if args and args[0] == "git" and len(args) > 1:
            invoked_subcommands.append(args[1])
        return real_run(args, *a, **kw)

    monkeypatch.setattr(
        "treasury_auction_stress.governance.history_audit.subprocess.run", spy_run
    )
    audit_history(repo)

    assert invoked_subcommands, "expected the audit to invoke git at least once"
    assert not (set(invoked_subcommands) & forbidden), invoked_subcommands


def test_enumerate_reachable_blobs_dedupes_and_reports_size(repo: Path):
    _commit(repo, "a.txt", "same content\n", "add a")
    _commit(repo, "b.txt", "same content\n", "add b, identical bytes to a")

    blobs = enumerate_reachable_blobs(repo)
    matching = [b for b in blobs.values() if set(b.paths) >= {"a.txt", "b.txt"}]
    assert len(matching) == 1  # one blob sha, two paths
    assert matching[0].size == len(b"same content\n")


def test_history_finding_representation_never_contains_the_canary_value(repo: Path):
    """End-to-end no-leak guarantee at the HistoryFinding layer: neither
    the dataclass fields nor its repr/str contain the canary secret or
    home path anywhere."""
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")
    _commit(
        repo,
        "report.md",
        f"ran from {CANARY_HOME_PATH}/.venv/bin/python3\n",
        "record local run",
    )

    result = audit_history(repo)

    assert result.findings
    for f in result.findings:
        rendered = repr(f)
        assert CANARY_SECRET not in rendered
        assert CANARY_HOME_PATH not in rendered
        assert CANARY_SECRET not in f.description
        assert CANARY_HOME_PATH not in f.description
