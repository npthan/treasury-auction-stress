"""Offline, network-free checks used to decide whether a file, the
committed Phase 8 dashboard aggregate, or the full reachable git history
(see `history_audit.py`) is safe for a public repository.

Every function here is regex/structure-based, not a live secret
scanner, and every function is designed to never surface the actual
matched substring of a suspected secret OR of identifying information
(home paths, emails, session ids) -- only its category, the location it
was found at, a redacted description, and (for identifying-info matches)
a non-reversible fingerprint. This mirrors the same discipline
`docs/point_in_time_rules.md` and the project's other modules apply to
their own domain: verifiable, reusable, and testable rather than a
one-off terminal command.

Disposition policy: a `Finding.severity` of 'hard_block' fails
`audit_tree`; consent to publish a hard-blocking identifying-info match
is never inferred from its mere presence in a tracked file (including a
project's own `authors` email in pyproject.toml) -- only an explicit,
per-value fingerprint allowlist can permit one. See
`_severity_for_category` and `audit_tree`'s docstring.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_header": re.compile(
        r"-----BEGIN (RSA|DSA|EC|OPENSSH|PGP)? ?PRIVATE KEY-----"
    ),
    "bearer_token": re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{20,}"),
    "generic_assignment": re.compile(
        r"(?i)\w*(?:api[_-]?key|password|secret|token)\w*\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_\-/+=.]{8,}['\"]?"
    ),
    "fred_api_key_env": re.compile(r"\bFRED_API_KEY\b\s*[:=]\s*['\"a-zA-Z0-9]"),
}

_IDENTIFYING_INFO_PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute_home_path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.\-]+"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ipv4_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "dev_session_url": re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)+/code/session_[A-Za-z0-9]+"),
    "home_dotdir_path": re.compile(r"~/\.[A-Za-z0-9][A-Za-z0-9_-]*/"),
}

# Disposition policy (Phase 9 acceptance-review remediation): a finding's
# category decides whether it can ever pass silently.
#
#   hard_block -- must fail the audit. Real credentials/keys, session or
#     auth material, non-consensual local usernames/home paths, emails,
#     and any other private absolute path. Consent to publish this is
#     NEVER inferred merely because the value appears in a tracked file
#     or in git history -- including a project's own `authors` email in
#     pyproject.toml. The only way a hard_block category stops failing
#     the audit for a *specific* value is a caller-supplied, explicit
#     allowlist (see `audit_tree`'s `allowed_identifying_info` param) --
#     never a bare directory/path exemption.
#   review -- lower-confidence signal (a bare IPv4 that might be a public
#     example address; a home-directory dotfile-path mention that might
#     be documentation prose, not a real path) that a human should look
#     at but that does not by itself fail the audit.
#   info -- collected for completeness only (currently unused; reserved
#     for future low-signal categories).
_HARD_BLOCK_IDENTIFYING_CATEGORIES = frozenset(
    {
        "identifying_info:absolute_home_path",
        "identifying_info:email_address",
        "identifying_info:dev_session_url",
    }
)
_REVIEW_IDENTIFYING_CATEGORIES = frozenset(
    {
        "identifying_info:ipv4_address",
        "identifying_info:home_dotdir_path",
    }
)


def _severity_for_category(category: str) -> str:
    """Map a Finding category to 'hard_block', 'review', or 'info'."""
    if category == "rejected_path" or category.startswith("secret:"):
        return "hard_block"
    if category in _HARD_BLOCK_IDENTIFYING_CATEGORIES:
        return "hard_block"
    if category in _REVIEW_IDENTIFYING_CATEGORIES:
        return "review"
    return "info"


# Directories whose real (non-placeholder) contents must never be
# published; mirrors .gitignore's data/{raw,interim,processed} rule.
_BULK_DATA_DIR_PREFIXES = ("data/raw/", "data/interim/", "data/processed/")
_PLACEHOLDER_NAMES = {".gitkeep"}

_REJECTED_SUFFIXES = (
    # Bulk/binary data formats
    ".parquet",
    ".db",
    ".sqlite",
    ".sqlite3",
    # Serialized models / pickled objects
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".onnx",
    ".joblib",
    ".npz",
    # Archives
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    # Private key / certificate material
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    # Compiled/executable binaries
    ".exe",
    ".dll",
    ".dylib",
    ".so",
)
_REJECTED_EXACT_OR_PREFIX = (
    ".env",  # exact
)
_REJECTED_PATH_FRAGMENTS = (
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".ipynb_checkpoints/",
    ".venv/",
    "/.venv/",
)
_REJECTED_BASENAMES = {".DS_Store"}

# Narrow allowlist for tests/fixtures/: small, plain-text, hand-picked
# sample data and documentation only. This does NOT exempt fixtures from
# any rejection check above (suffix/basename/fragment/bulk-data-dir) --
# it only additionally restricts fixtures to these formats, and a
# fixture file must still be under the size cap and clean of secret
# patterns (enforced by `audit_tree`, which also runs the content scans).
#
# Deliberately does NOT include .xlsx. An earlier version of this
# allowlist carried a disclosed .xlsx exception for two small, real
# Philadelphia Fed RTDSM excerpts -- those files were removed in the
# Phase 9 acceptance-review remediation (replaced by wholly synthetic,
# Python-generated fixture bytes; see tests/rtdsm_synthetic_fixtures.py)
# specifically because RTDSM's public-redistribution status is
# unresolved. Leaving .xlsx off this list is defense in depth: if a real
# (or any) xlsx workbook is ever added back under tests/fixtures/, the
# audit rejects it by format rather than silently allowing it again.
_FIXTURE_ALLOWED_SUFFIXES = {
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".yml",
    ".yaml",
    ".py",
}
_FIXTURE_MAX_BYTES = 512_000


@dataclass(frozen=True)
class Finding:
    """One audit finding. `description` must never embed the actual
    matched secret/PII substring -- only a redacted, human-readable
    note. `fingerprint` (for identifying-info matches) is a
    non-reversible sha256-derived id (`_fingerprint`) that identifies a
    specific matched value without making it recoverable -- combined
    with the finding's path into a path-scoped token (`allowlist_token`),
    it is the only handle an explicit allowlist (`audit_tree`'s
    `allowed_identifying_info`) can use to un-block one specific value
    at one specific path.

    `severity` is one of 'hard_block', 'review', or 'info' -- see
    `_severity_for_category`. Callers must never construct a Finding
    with a `description` built from an unredacted regex match.
    """

    category: str
    location: str
    description: str
    severity: str = "info"
    fingerprint: str | None = None


def _redact(match: str, keep: int = 2) -> str:
    """Never used to expose a real secret in a Finding -- only to build
    a length-preserving redacted preview if a caller ever wants one.
    """
    if len(match) <= keep * 2:
        return "*" * len(match)
    return match[:keep] + "*" * (len(match) - keep * 2) + match[-keep:]


def _fingerprint(value: str) -> str:
    """Non-reversible short fingerprint of a matched value: identifies
    and deduplicates a finding without ever exposing or making the
    original value recoverable. Never embed `value` itself in output.
    """
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()
    return f"fp:{digest[:12]}"


def finding_path(f: Finding) -> str:
    """The repo-relative path a finding refers to, with the trailing
    ':<lineno>' that content-scan locations carry stripped off."""
    path, sep, tail = f.location.rpartition(":")
    if sep and tail.isdigit():
        return path
    return f.location


def allowlist_token(path: str, fingerprint: str) -> str:
    """The path-scoped allowlist token for one reviewed finding:
    '<repo-relative-path>::<fp:...>'. An allowlist entry permits one
    specific reviewed value at one specific path only -- the SAME value
    appearing at any other path still hard-blocks, so an approved
    canary can never be quietly reused elsewhere in the tree or
    history."""
    return f"{path}::{fingerprint}"


def scan_for_secret_patterns(text: str, location: str = "<text>") -> list[Finding]:
    """Regex-based secret/credential scan. Returns one Finding per
    matched category per line, with the actual matched value redacted
    out of the description entirely (category + line number only) but
    fingerprinted the same way identifying-info matches are -- so a
    specific, individually-reviewed known-fake value (e.g. a test
    suite's own deliberately-fake canary credential) can be permitted
    via `audit_tree`'s `allowed_identifying_info`, without weakening the
    check for any other, unreviewed value.
    """
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for category, pattern in _SECRET_PATTERNS.items():
            m = pattern.search(line)
            if m:
                findings.append(
                    Finding(
                        category=f"secret:{category}",
                        location=f"{location}:{lineno}",
                        description=(
                            f"line {lineno} matches the '{category}' pattern; "
                            "value redacted"
                        ),
                        severity="hard_block",
                        fingerprint=_fingerprint(m.group(0)),
                    )
                )
    return findings


def scan_for_identifying_info(text: str, location: str = "<text>") -> list[Finding]:
    """Regex-based scan for local-machine paths, emails, IPs, and
    development-tool session artifacts. The matched value is NEVER embedded
    in the returned Finding -- only its category, line number, and a
    non-reversible sha256-derived fingerprint (`_fingerprint`), which
    identifies/dedups a specific match without making it recoverable.
    """
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for category, pattern in _IDENTIFYING_INFO_PATTERNS.items():
            m = pattern.search(line)
            if m:
                full_category = f"identifying_info:{category}"
                fp = _fingerprint(m.group(0))
                findings.append(
                    Finding(
                        category=full_category,
                        location=f"{location}:{lineno}",
                        description=(
                            f"line {lineno} matches '{category}' ({fp}); value not shown"
                        ),
                        severity=_severity_for_category(full_category),
                        fingerprint=fp,
                    )
                )
    return findings


def is_publishable_path(path: str) -> bool:
    """True if `path` (a repo-relative, forward-slash path) is allowed
    in a public-release allowlist. Rejects bulk data, .env files, and
    common cache/venv/checkpoint/OS-metadata/serialized-model/archive/
    key-material/executable paths -- these checks apply uniformly to
    every path, including `tests/fixtures/`, which gets NO blanket
    exemption. `tests/fixtures/` is additionally restricted to a narrow
    set of plain-text formats (`_FIXTURE_ALLOWED_SUFFIXES`): a fixture
    passing every rejection check above still fails here if its suffix
    isn't on that allowlist (e.g. an extensionless binary). `.gitkeep`
    placeholders are always allowed, anywhere.
    """
    normalized = PurePosixPath(path.replace("\\", "/"))
    posix = str(normalized)
    basename = normalized.name
    suffix = normalized.suffix.lower()

    if basename in _PLACEHOLDER_NAMES:
        return True

    if any(posix.startswith(prefix) for prefix in _BULK_DATA_DIR_PREFIXES):
        return False
    if basename == ".env" or basename.startswith(".env."):
        return False
    if basename in _REJECTED_BASENAMES:
        return False
    if any(fragment in f"/{posix}" for fragment in _REJECTED_PATH_FRAGMENTS):
        return False
    if suffix in _REJECTED_SUFFIXES:
        return False

    if posix.startswith("tests/fixtures/"):
        return suffix in _FIXTURE_ALLOWED_SUFFIXES

    return True


def is_publishable_fixture_size(path: str, size_bytes: int) -> bool:
    """True unless `path` is under tests/fixtures/ and exceeds the
    narrow fixture size cap (`_FIXTURE_MAX_BYTES`) -- fixtures are meant
    to be small, hand-picked excerpts, not bulk data. No-op (always
    True) for paths outside tests/fixtures/, which are governed by
    `find_large_tracked_files`'s general threshold instead.
    """
    posix = str(PurePosixPath(path.replace("\\", "/")))
    if not posix.startswith("tests/fixtures/"):
        return True
    return size_bytes <= _FIXTURE_MAX_BYTES


def check_dashboard_json_safe(json_obj: object) -> list[Finding]:
    """Recursively inspect a parsed JSON object (e.g. the Phase 8
    dashboard aggregate) for keys/values that look like a secret, a
    local filesystem path, or an unexpectedly granular per-record PII
    field. Aggregate metrics, model ids/labels, content-hash digests,
    and public auction facts (tenor, date, auction key, error figures)
    are all expected and not flagged.
    """
    findings: list[Finding] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}.{key}"
                findings.extend(scan_for_secret_patterns(str(value), child_path))
                if isinstance(value, str):
                    findings.extend(scan_for_identifying_info(value, child_path))
                walk(value, child_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(json_obj, "$")
    return findings


def find_large_tracked_files(
    tracked_paths: list[str], root: Path, threshold_bytes: int = 1_000_000
) -> list[tuple[str, int]]:
    """Given a list of git-tracked repo-relative paths and the repo
    root, return (path, size_bytes) for every file at or above
    `threshold_bytes`, sorted largest first. Reporting only -- never
    deletes or modifies anything.
    """
    oversized: list[tuple[str, int]] = []
    for rel_path in tracked_paths:
        full_path = root / rel_path
        try:
            size = full_path.stat().st_size
        except OSError:
            continue
        if size >= threshold_bytes:
            oversized.append((rel_path, size))
    oversized.sort(key=lambda item: item[1], reverse=True)
    return oversized


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    findings: list[Finding] = field(default_factory=list)


def audit_tree(
    paths_and_contents: list[tuple[str, str]],
    allowed_identifying_info: frozenset[str] = frozenset(),
) -> AuditResult:
    """Compose the path-allowlist and secret/identifying-info scans
    over a list of (repo_relative_path, file_text_content) pairs.

    `passed=False` if ANY finding has `severity == "hard_block"`:
    a rejected path, a matched secret pattern, or a hard-blocking
    identifying-info category (absolute home path, email address,
    development-tool session URL -- see `_HARD_BLOCK_IDENTIFYING_CATEGORIES`).
    Consent to publish a hard-blocking match is never inferred from its
    mere presence -- the ONLY way to keep a specific match from failing
    the audit is to list its path-scoped allowlist token (see
    `allowlist_token`; e.g. `"tests/x.py::fp:abc123abc123"`) in
    `allowed_identifying_info`, an explicit, caller-supplied, per-value,
    per-path policy decision, never a blanket category, directory, or
    value-anywhere exemption. The same value at a different path is NOT
    covered by an existing entry and still fails the audit. This applies to `secret:*` findings too (a
    project's own test suite legitimately needs a reviewed way to
    exempt its deliberately-fake test canaries -- see
    `configs/governance_audit_allowlist.yml`) -- but NOT to
    `rejected_path` findings, which have no fingerprint (`None`) since
    there is no single value to individually review. `review`-severity
    findings (e.g. a bare IPv4) are always collected but never fail the
    audit on their own.
    """
    all_findings: list[Finding] = []

    for path, content in paths_and_contents:
        if not is_publishable_path(path):
            all_findings.append(
                Finding(
                    category="rejected_path",
                    location=path,
                    description=f"'{path}' is not on the public-release allowlist",
                    severity="hard_block",
                )
            )

        all_findings.extend(scan_for_secret_patterns(content, location=path))
        all_findings.extend(scan_for_identifying_info(content, location=path))

    passed = not any(
        f.severity == "hard_block"
        and (
            f.fingerprint is None
            or allowlist_token(finding_path(f), f.fingerprint)
            not in allowed_identifying_info
        )
        for f in all_findings
    )
    return AuditResult(passed=passed, findings=all_findings)
