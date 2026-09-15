"""Public-release governance tooling (Phase 9, with acceptance-review
remediation): reusable, offline, network-free checks for whether a file,
the committed dashboard aggregate, or the full reachable git history is
safe to publish. No function in this subpackage ever prints or returns a
matched secret's or identifying-info value's actual content -- only its
category, location, and (for identifying info) a non-reversible
fingerprint. See `release_audit.py` (current tree) and
`history_audit.py` (full git history); `audit_cli.py` runs both.
"""
