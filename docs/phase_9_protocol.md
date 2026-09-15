# Phase 9 protocol (frozen before any Phase 9 correction)

**Superseded by the Phase 9 acceptance review.** An independent
acceptance review found the original Phase 9 self-report's verdict
unsupportable (a sensitive git-history object was not actually removed,
among other findings) and performed a user-directed remediation pass.
Read the acceptance review for the current, authoritative status. This
file (and its machine-readable companion,
`configs/phase_9_release.yml`, which has a syntax-only-fix addendum at
its end) is kept as the original frozen criteria for historical/
evidentiary reasons, unedited below this notice.

This is the human-readable companion to `configs/phase_9_release.yml`.
Both were written and committed to disk *before* any Phase 9 audit
finding was corrected, per this project's own rule of freezing
acceptance criteria before making the changes they will judge. If a
criterion below turns out to be the wrong criterion, that will be
recorded as a disclosed note in the Phase 9 release audit
— this file itself will not be quietly edited to make a later verdict
pass.

## Baseline (recorded at Phase 9 start)

- Branch: `main`
- HEAD commit: `222cee3` — "Complete Phase 8 interpretation, dashboard,
  and research report"
- Working tree: clean (`git status` — nothing to commit)
- `git remote -v`: no remote configured
- Phase 8 status: **ACCEPTED** (the Phase 8 acceptance review)
- Offline lint: `uv run ruff check .` — all checks passed
- Baseline SHA-256 checksums of every accepted Phase 7 prediction/metric
  parquet and every committed Phase 8 report/dashboard artifact were
  recorded before any Phase 9 file was touched (see
  the Phase 9 release audit Section 1 for the values).

## What Phase 9 is and is not allowed to do

Allowed: narrowly-scoped corrections to a *proven* security, privacy,
licensing, reproducibility, documentation, or packaging defect, each one
documented and regression-tested. Everything else in this phase is
audit, documentation, and reporting — no modeling code, no target
formula, no fold construction, no accepted metric changes.

Not allowed, under any framing: new modeling research; retraining or
tuning; adding FRED/ALFRED or any new data source; requiring an API key
for ordinary tests/dashboard use; downloading new bulk data; exposing
gitignored local data; installing a scanner without explicit permission;
choosing a license on the project owner's behalf; fake GitHub URLs/badges;
rewriting git history; or running `git add`/`commit`/`push`/any
destructive git command.

## 9A — Tree and history safety audit

Read `configs/phase_9_release.yml`'s `phase_9a_tree_and_history_audit`
block for the exact checklist. In short: scan the current tracked tree
and every reachable git commit for secrets, credentials, PII, bulk
data, and unexpected binaries; verify `.gitignore` coverage; assess the
committed Phase 8 dashboard JSON's content; inventory large git objects.
No secret value is ever printed. No `git gc`/prune/rewrite is run. A
found secret or prohibited dataset in history halts public-release prep
for that item and is reported by file/commit/category only.

## 9B — Data provenance, redistribution, and licensing

For each of the five ingested sources (Treasury Fiscal Data auctions, NY
Fed Primary Dealer Statistics, Treasury daily par yield curve, CFTC TFF
positioning, Philadelphia Fed RTDSM), re-check the *current* official
primary-source terms and document publisher, what's downloaded/derived,
whether raw data is committed, fixture provenance, attribution/
redistribution terms, dashboard/report suitability for public release,
the date checked, and any open uncertainty. Dependency licenses are
reviewed from installed package metadata. The project software license
is **not** chosen by this phase — it is recorded as an open decision for
the project owner, with a brief MIT recommendation.

## 9C — Clean-room reproducibility

Verify the accepted Phase 8 commit from a fresh temporary clone outside
this working directory (a narrow temp dir, never a broad user
directory; nothing in the original repo is touched). Checks: Python
3.12 constraint, `uv sync --locked`, offline tests, ruff, dashboard HTTP
smoke test from the repo root, no credential requirement for default
tests, live-network tests opt-in only, byte-identical Phase 8 artifacts
across two runs, unchanged Phase 7 checksums, and that documented
commands work copy-pasted verbatim.

## 9D — Minimal public-repository polish

README gets the specific required sections listed in
`configs/phase_9_release.yml` (research question, honest result,
mandatory caveats, quick start, test/lint commands, dashboard preview,
cross-links, layout section, public-vs-ignored-data distinction,
non-stale phase status) and loses local paths, stale dates, and
internal planning language. A GitHub Actions workflow is added only if
it can run with zero credentials and zero private data, pinned to
current official actions, read-only permissions, offline tests only.

## 9E — Public-release checklist and verdict

the Phase 9 release audit is produced with the 18
required sections and one of exactly two verdicts:
`READY FOR PHASE 9 ACCEPTANCE REVIEW` (only if the sole remaining item
is an explicitly-named user decision such as the license) or `BLOCKED`
(any unresolved secret, sensitive history object, redistribution
problem, failed clean-room check, or other material risk). Phase 9 is
never self-accepted; it always remains pending a separate acceptance
review, matching every earlier phase's pattern in this project.
