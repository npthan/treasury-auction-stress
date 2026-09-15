# Project engineering rules

This file is the durable checklist of rules that govern all work in
this repository. It consolidates the standing rules established during
Phase 0/1 setup and the additional evaluation-discipline rules adopted
with the Phase 5–7 task specifications; see `docs/` for
the full reasoning behind each one. Source code and reports cite these
rules by name (for example "the no-fabricated-results rule").

## Environment

- Python **3.12** only (`requires-python = ">=3.12,<3.13"` in
  `pyproject.toml`, pinned via `.python-version`). Do not widen this
  range or let a tool auto-select 3.13/3.14.
- Use **uv** for everything: `uv sync`, `uv run <cmd>`, `uv add
  <package>`. Never manually activate `.venv` — always run commands
  through `uv run ...`.
- Lint with `uv run ruff check .`; test with `uv run pytest`.
- Do not add a new dependency without a clear, demonstrated need;
  formatting convenience alone does not qualify.

## Repository structure

- Reusable code lives under `src/treasury_auction_stress/`, organized
  into `data/`, `features/`, `models/`, `evaluation/`,
  `visualization/` subpackages. Notebooks, if any, are for exploration
  and presentation only — no reusable logic may live only in a
  notebook.
- Raw downloads go in `data/raw/`, intermediate outputs in
  `data/interim/`, final analysis-ready tables in `data/processed/`.
  All three are gitignored; only `.gitkeep` placeholders are tracked.
- Small, hand-picked test fixtures go in `tests/fixtures/` and ARE
  committed — this is the one exception to "don't commit data."

## Raw-data immutability

- Once written, a raw downloaded file must never be mutated in place.
  A new pull writes a new file (deterministic/content-aware naming) or
  is skipped if an identical cached artifact already exists.
- Every raw artifact gets a metadata sidecar recording source, exact
  request URL and parameters, retrieval timestamp, HTTP status,
  response format, row count, date range, checksum, and schema/code
  version.

## Point-in-time discipline

- See `docs/point_in_time_rules.md` for the full rules. In short:
  observation date, publication date, retrieval date, and prediction
  cutoff are four different things — never substitute one for another.
- Auction-result fields are never features for a pre-auction model.
- Never assume `auction_date < fold year` alone proves a result was
  safely available by a fold's fit origin — availability must be
  established by an explicit `result_safe_available_date <=
  fold_fit_origin` comparison, never inferred from the auction date.
- Any scaler, encoder, imputer, threshold, feature-selection step, or
  residualization must be fit on training-fold data only, refit
  per chronological fold — never fit once on the full dataset.
- Evaluation is chronological rolling-origin. Auctions are never
  shuffled randomly across train/test.
- **Phase 5 rule**: any combined feature matrix must be assembled by
  *whitelisting* the exact predictor columns declared in
  `configs/phase_5_features.yml` (see
  `treasury_auction_stress.features.feature_manifest`), never by
  blacklisting known-bad columns from a wider join. A blacklist can
  miss a renamed field; a whitelist cannot accidentally include one
  that was never asked for. Keep the contract and the actual matrix
  columns in sync — `feature_matrix.validate_predictor_matrix` enforces
  this automatically.
- **Same-day announcements are never ordered relative to each other.**
  Multiple auctions announced on the identical calendar date (routine
  in this project's real data — batched multi-tenor announcements, and
  same-tenor reopening/new-issue pairs) must never have one treated as
  preceding another for a "previous auction" or "trailing window"
  feature. Use `merge_asof(..., allow_exact_matches=False)` or an
  aggregate-then-shift(1) pattern, never a plain `.shift()` over a
  same-day-ties-broken-by-row-order sort — see
  `treasury_auction_stress.features.auction_candidate_features`'s
  module docstring for the established pattern.

## Evaluation and reporting discipline

- Never optimize the evaluation protocol to flatter a model. A metric
  that would be leaky or misleading is not reported at all rather than
  reported with caveats.
- Secondary targets evaluated under the same rigorous protocol need
  target-appropriate units: share targets are reported in percentage
  points, `bid_to_cover_ratio` in its own native ratio units — the two
  must never be mixed in one table.
- Skill versus a fold's own mean and skill versus a pre-specified
  strong baseline are different quantities; conflating the two R²
  notions is prohibited, and both are named explicitly wherever either
  appears.
- Every evaluation must report exactly which rows were attempted
  versus actually scored, and why any were excluded — the evaluated
  subset must never vary silently per model.

## Testing

- Every pipeline component needs tests that don't depend on live
  network access — use fixtures in `tests/fixtures/`. Live-network
  tests, if added, must be clearly separated (e.g. a marker) so the
  default `pytest` run stays deterministic and offline.
- Don't delete a test to make a failure go away; fix the underlying
  code or, if the test's assumption was wrong, fix the assumption and
  say so.

## No fabricated results

- Never report row counts, date ranges, or metrics that were not
  actually produced by running the pipeline/tests. If live data could
  not be fetched, say so explicitly and mark any fixture-derived
  numbers as fixture output, not live results.

## Secrets and credentials

- Never commit `.env`, API keys, or any credential. Don't add a new
  credential-requiring source without explicit review of how that
  credential is obtained and stored.

## Source preference

- Prefer official, primary sources (see `configs/sources.yml`). Don't
  substitute an unofficial aggregator, and don't ingest a source
  marked `authorized_in_current_phase: false` until its phase arrives.
- Don't assume a field exists — verify it against the actual API
  response/data dictionary before writing code that depends on it.
