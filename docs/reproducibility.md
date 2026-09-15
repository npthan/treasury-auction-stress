# Reproducibility guide

This document explains how to set up this project from a clean checkout
and how to verify it independently, without live network access or any
API key. Every command and result below was actually run — originally
during Phase 9 against a fresh local clone of the accepted commit, and
re-run/extended during the Phase 9 acceptance review (which added the
git-free sanitized-export scenario) — in isolated temporary directories
outside the working repository (never a broad user directory), each
deleted afterward. The original repository's `.git` history was never
modified by any of these checks.

## Setup (Python 3.12 + uv)

```bash
uv python install 3.12   # if not already installed
uv sync --locked         # install exactly the locked dependency versions
```

`uv sync --locked` succeeded in the clean-room clone. Disclosed honestly:
this ran against this machine's existing local `uv` package cache, not a
network-fresh empty cache — it verifies the lockfile resolves and
installs correctly, not that every package is fetchable from a cold
cache with no network at all.

You should not need to manually activate `.venv` — every command below
uses `uv run <command>`, which runs inside the project's environment
automatically.

## Offline verification

```bash
uv run pytest      # the default offline test suite
uv run ruff check . # linting
```

**What to expect, and why the numbers differ across three different
checkouts** (all re-verified during the Phase 9 acceptance review, not
carried over from an earlier claim):

1. **This project's own working repository** (real, locally-downloaded
   pipeline data under the gitignored `data/raw/`/`data/processed/`
   directories from earlier phases, plus the Phase 9 governance code
   and tests, which are not yet committed but present on disk):
   `uv run pytest -q` reports **743 passed, 11 deselected**.
2. **A genuine `git clone` of the accepted `222cee3` commit** (Phase
   0-8 only — the Phase 9 governance code and tests are not committed,
   so they don't exist in this clone at all; no `data/raw/`,
   `data/interim/`, or `data/processed/` content beyond the `.gitkeep`
   placeholders): `uv run pytest -q` reports **608 passed, 60 skipped,
   8 deselected** (676 collected). The 60 skips are tests that
   conditionally exercise real local pipeline outputs when present and
   skip gracefully (not fail) when absent.
3. **A git-free "sanitized export"** — every file that would currently
   be tracked after `git add -A` (Phase 0-9, including the governance
   code/tests), copied to a fresh directory with no `.git` at all (see
   the Phase 9 acceptance review's sanitized-export section):
   `uv run pytest -q` reports **679 passed, 60 skipped, 4 failed, 11
   deselected** (754 collected). The 4 failures are exactly the 4 new
   governance tests that shell out to `git ls-files` to compare the
   tree/history against a live repository — they cannot run without a
   `.git` directory at all, which this deliberately git-free export
   doesn't have. This is expected and not a code defect: it's what
   requirement F's "without creating a Git repository" produces for any
   test that specifically exercises git plumbing.

`uv run ruff check .` reports "All checks passed" in all three cases.

The `live_network`-marked tests are excluded by default via
`pyproject.toml`'s `[tool.pytest.ini_options]` `addopts`. The default
`uv run pytest` invocation requires no network access and no credential
or environment variable of any kind — none of this project's five data
sources need an API key.

## Optional: live-data commands

Only run these if you want to exercise real network calls against the
five public data sources (still no API key needed for any of them):

```bash
# Run the live-network-marked tests explicitly:
uv run pytest -o addopts="" -m live_network

# Re-run a data source's own ingestion CLI (writes into gitignored
# data/raw/; each CLI's --help documents its options):
uv run python -m treasury_auction_stress.data.cli
```

As of the Phase 9 acceptance review this includes three RTDSM tests
(`tests/test_live_network.py`) that verify real, historically documented
facts (a genuine 2010 unemployment-rate revision, the COVID-19
industrial-production collapse, an October 2025 reporting gap) against
live-fetched data — the always-run offline suite's RTDSM tests use
wholly synthetic fixtures instead (`tests/rtdsm_synthetic_fixtures.py`),
since the two real `.xlsx` excerpts previously committed here were
removed (see `docs/data_sources_and_licensing.md` Section 5). All 11
`live_network` tests were actually run and passed during this review.

## Dashboard preview

From the **repository root** (not from inside `dashboard/` — serving
from `dashboard/` itself works for the dashboard's own data but 404s
its "Project README" link, which lives one level up):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/dashboard/
```

Verified in both the clean-room clone AND the git-free
sanitized export (see Section above): `GET /dashboard/`, `GET
/dashboard/data/phase8_dashboard_data.json`, and `GET
/README.md` all returned **HTTP 200**,
using only the three `.gitkeep`-only `data/` directories — the
dashboard's public/demo mode needs no locally-downloaded or -processed
data, and no `.git` directory either.

## What's generated vs. what's committed

**Committed** (present in any clone): all source code under `src/`
(including the `governance/` release-audit and full-history-audit
tooling), all tests under `tests/`, including small, real, hand-picked
excerpts in `tests/fixtures/` for four of the five data sources — RTDSM
is the exception: its fixtures are wholly synthetic, generated in Python
at test time (`tests/rtdsm_synthetic_fixtures.py`), not committed
binary files — plus `configs/`, `docs/`, the static
dashboard (`dashboard/index.html`/`app.js`/`style.css`) and its small
generated aggregate `dashboard/data/phase8_dashboard_data.json`, and
`LICENSE` (MIT, this project's own code/docs only).

**Gitignored, never committed** (present only after you run the
pipeline locally): everything under `data/raw/`, `data/interim/`, and
`data/processed/` except the `.gitkeep` placeholders, and `.venv/`. A
fresh clone therefore gives you the code, tests, documentation, and the
small committed aggregates — not a one-command full pipeline rerun.
Regenerating `data/processed/phase_7_*.parquet` or
`dashboard/data/phase8_dashboard_data.json` from scratch requires first
running the live ingestion pipeline against all five real external
sources (each phase's own CLI under `treasury_auction_stress`), which
this project does not do automatically in CI (see below) or in this
guide, to avoid requiring network access for ordinary setup/testing.

## Generated reports and figures: the untracked `artifacts/` directory

Generated Markdown reports and PNG figures (data-quality profiles, the
feature matrix / leakage-audit / feature-dictionary reports, the Phase
6/7 results and protocol reports, and the Phase 8 interpretation
report) are **intentionally not tracked** in this repository. Every
generation CLI writes them to a gitignored local `artifacts/` directory
by default, creating it on demand — a fresh clone has no `artifacts/`
directory until a generation command runs, and nothing under it is ever
committed. Each CLI accepts `--reports-dir <path>` to write elsewhere.
All report/figure writes are fail-before-write and atomic (a failing
validation gate leaves previously-written outputs untouched), e.g.:

```bash
# Regenerate the Phase 6 reports and figures into artifacts/ (requires
# the gitignored data/processed/ pipeline outputs to exist locally):
uv run python -m treasury_auction_stress.evaluation.phase6_cli
# or into a directory of your choice:
uv run python -m treasury_auction_stress.evaluation.phase6_cli --reports-dir /tmp/out
```

## Troubleshooting

- **`uv sync --locked` fails**: this means your environment cannot
  resolve the exact versions pinned in `uv.lock` (e.g. no network access
  and an empty local cache, or an incompatible platform). Try `uv sync`
  without `--locked` to let `uv` re-resolve, but note this may install
  different versions than what this project was tested against.
- **A `live_network`-marked test is unexpectedly running**: the default
  `uv run pytest` should never select these — if you see one run, check
  you didn't pass `-m live_network` or clear `addopts` accidentally.
- **The dashboard's report links 404**: you likely started the preview
  server from inside `dashboard/` instead of the repository root — see
  "Dashboard preview" above.
- **`phase8_cli`/`phase7_cli`/etc. fail on a fresh clone**: these read
  already-generated `data/processed/*.parquet` files from earlier
  phases, which are gitignored and not committed — you would need to run
  the earlier phases' pipelines first, against live data, to produce
  them locally.
