"""End-to-end tests for the governance audit CLI
(`treasury_auction_stress.governance.audit_cli`), run against small,
throwaway synthetic git repositories -- never this project's own repo.

Proves finding 3 (no leaked match) at the outermost layer: the actual
captured CLI stdout, not just an in-process Finding object.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CANARY_SECRET = "AKIACANARY0123456789"
CANARY_EMAIL = "definitely-not-a-real-person-9f8e7d@example-canary.test"


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


def _commit(repo: Path, rel_path: str, content: str, message: str) -> None:
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    _git(repo, "add", rel_path)
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _init_repo(tmp_path)


def _run_cli(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "treasury_auction_stress.governance.audit_cli"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_reports_blocked_and_never_prints_the_canary_secret(repo: Path):
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")

    result = _run_cli(repo)

    assert result.returncode == 1
    assert "BLOCKED" in result.stdout
    assert CANARY_SECRET not in result.stdout
    assert CANARY_SECRET not in result.stderr


def test_cli_reports_blocked_for_modified_but_not_deleted_secret_in_history(repo: Path):
    _commit(repo, "config.py", f"AWS_KEY = '{CANARY_SECRET}'\n", "add key")
    _commit(repo, "config.py", "AWS_KEY = '<redacted>'\n", "redact key")

    result = _run_cli(repo)

    assert result.returncode == 1
    assert "Full-history audit" in result.stdout
    assert "BLOCKED" in result.stdout
    assert CANARY_SECRET not in result.stdout
    assert CANARY_SECRET not in result.stderr


def test_cli_never_prints_a_canary_email_anywhere_in_output(repo: Path):
    _commit(repo, "pyproject.toml", f'authors = [{{email = "{CANARY_EMAIL}"}}]\n', "add authors")

    result = _run_cli(repo)

    assert result.returncode == 1
    assert CANARY_EMAIL not in result.stdout
    assert CANARY_EMAIL not in result.stderr


def test_cli_passes_on_a_clean_synthetic_repo(repo: Path):
    _commit(repo, "README.md", "# hello\nnothing sensitive here\n", "init")

    result = _run_cli(repo)

    assert result.returncode == 0
    assert "OVERALL: PASSED" in result.stdout


def test_cli_allow_fingerprint_flag_permits_one_explicit_value(repo: Path):
    """The --allow-fingerprint flag takes a path-scoped
    '<repo-relative-path>::fp:<hex>' token and applies to both the
    current-tree and full-history scans (see
    test_governance_audit_allowlist.py for the
    token-symmetric behavior at the audit_history() level). This
    scenario commits CLEAN content first (history stays PASSED
    throughout) and only adds the canary to the uncommitted working
    tree -- the realistic case of "fix a working-tree file before
    committing it" -- so history has nothing to allowlist here in the
    first place."""
    _commit(repo, "pyproject.toml", 'authors = [{name = "Someone"}]\n', "clean initial commit")
    (repo / "pyproject.toml").write_text(f'authors = [{{email = "{CANARY_EMAIL}"}}]\n')

    blocked = _run_cli(repo)
    assert blocked.returncode == 1
    assert "BLOCKED" in blocked.stdout

    # Extract the fingerprint from the (redacted) blocked run's own output.
    fingerprint = next(
        token
        for line in blocked.stdout.splitlines()
        for token in line.replace("(", " ").replace(")", " ").split()
        if token.startswith("fp:")
    )

    allowed = subprocess.run(
        [
            sys.executable,
            "-m",
            "treasury_auction_stress.governance.audit_cli",
            "--allow-fingerprint",
            f"pyproject.toml::{fingerprint}",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0
    assert "OVERALL: PASSED" in allowed.stdout
    assert CANARY_EMAIL not in allowed.stdout
