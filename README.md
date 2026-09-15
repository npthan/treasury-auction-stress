# Treasury Auction Balance-Sheet Stress Forecaster

A point-in-time, leakage-aware research pipeline that forecasts U.S.
Treasury auction outcomes — built end to end from raw official data
sources through chronological backtesting, with a static local
dashboard and an auditable release process. This is a research project,
not a finished, deployed product.

## Research question

Using only information publicly available **before** a U.S. Treasury
auction, can we forecast the distribution of primary-dealer take-down
and identify auctions likely to require unusually heavy dealer
balance-sheet absorption?

This is a research project about auction mechanics and dealer
balance-sheet stress — **not** a Treasury-direction trading strategy.
It does not optimize for a Sharpe ratio. The research universe is
nominal coupon auctions (2y, 3y, 5y, 7y, 10y, 20y, 30y), new issues
and reopenings tracked separately, 2010 onward. Bills, TIPS, and
floating-rate notes are out of scope.

## Headline results (retrospective backtest, 2015–2025 complete years)

Pooled out-of-sample MAE for `primary_dealer_share` (percentage
points), chronological annual rolling-origin backtest, both prediction
cutoffs:

| Model | Announcement | Pre-auction |
|---|---|---|
| Recent-history baseline, frozen per test year | 5.104 | 5.104 |
| **Recent-history baseline, adaptive within-year** | **4.216** | **4.217** |
| Gradient-boosted-tree challenger | 7.009 | 6.938 |
| Empirical-Bayes shrinkage challenger | 14.011 | 14.011 |

**Neither complex challenger beat either recency baseline** — a
negative result stated plainly. The adaptive baseline beat the frozen
one in 9 of 11 complete years. Recency and safely-available
within-year updates mattered more than model complexity in this
sample. A 90% prediction-interval method achieved 92.2% pooled
coverage (23.0pp mean width), and a stress-event classifier passed its
data-sufficiency gate but scored **worse than a no-skill baseline** on
Brier score (0.135/0.119 vs. 0.059) — it is not a reliable stress
detector, and is reported as such. These headline numbers are pinned
by regression tests against the accepted prediction tables
(`tests/test_phase8_interpretation.py`).

Three caveats apply to every number above:

1. Dealer take-down (`primary_dealer_share`) is an auction-allocation
   measure and at most a **proxy** for dealer balance-sheet absorption —
   not a direct measure of dealers' final holdings, funding stress, or
   an auction's "success."
2. The stress classifier's probabilities are **poorly calibrated**
   (Brier roughly 2.0–2.3× worse than no-skill).
3. The ~92% pooled interval coverage hides real year-to-year drift
   (81.9%/75.0% in 2015/2016, 92.9%–100% in 2021–2025) — pooled
   coverage alone is not evidence of stable calibration.

This is a retrospective analysis — **not** a fresh holdout, causal
evidence, or a trading strategy.

## Repository layout

```
.github/workflows/            CI: offline lint + tests only, no credentials, no live data
configs/                      Source registry, feature/evaluation contracts, release policy
dashboard/                    Static local dashboard (index.html/app.js/style.css) + generated data
data/{raw,interim,processed}/ Pipeline outputs (gitignored; not committed)
docs/                         Rules, target spec, data dictionary, licensing, reproducibility
src/treasury_auction_stress/  Reusable package (data, features, models, evaluation, visualization, governance)
tests/                        Offline test suite; tests/fixtures/ holds small committed fixtures
artifacts/                    Generated reports and figures (gitignored; created on demand)
```

## Data sources and attribution

Five official sources are ingested: the **U.S. Treasury Fiscal Data —
Treasury Securities Auctions** dataset
(https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/),
**Federal Reserve Bank of New York Primary Dealer Statistics**, the
**U.S. Treasury daily par yield curve**, **CFTC Traders in Financial
Futures (TFF), Futures Only** positioning, and the **Federal Reserve
Bank of Philadelphia Real-Time Data Set for Macroeconomists (RTDSM)**
for point-in-time macro vintages. None requires an API key. FRED/ALFRED
is deliberately excluded — see `docs/data_source_governance.md`.
`configs/sources.yml` is the full registry with each source's
authorization status; `docs/data_sources_and_licensing.md` documents
each source's redistribution/attribution terms as last checked
(documentation of terms, not a legal opinion) and the dependency-license
review. `docs/data_dictionary.md` is the verified schema reference.

**No raw, interim, or processed dataset from any source is committed.**
Only small hand-picked test fixtures (`tests/fixtures/` — the RTDSM
fixtures are wholly synthetic) and the small generated aggregate
`dashboard/data/phase8_dashboard_data.json` are tracked.

## Point-in-time and leakage controls

Observation date, publication date, retrieval date, and prediction
cutoff are treated as four different things throughout
(`docs/point_in_time_rules.md`). Auction-result fields are never
features for a pre-auction model; every scaler/encoder/threshold is
fit on training folds only, refit per chronological fold; feature
matrices are assembled by whitelisting the exact columns declared in
`configs/phase_5_features.yml` (enforced by
`features.feature_manifest`); same-day announcements are never ordered
relative to each other; and a leakage audit runs as a mandatory gate
before any feature matrix is written. `docs/project_rules.md` is the
full engineering-rules checklist.

## Modeling and evaluation

A pre-registered baseline ladder (tenor means, recent-history rolling
means) plus Ridge/Elastic Net, gradient-boosted-tree, empirical-Bayes
shrinkage, residual-quantile interval, and logistic stress-event models
— all evaluated with annual, expanding-window, rolling-origin backtests
(initial history 2010–2014; complete-year test folds 2015–2025;
partial 2026 reported separately as provisional), at two cutoffs
(announcement and pre-auction). Evaluation protocols are frozen as
machine-readable configs (`configs/phase_6_evaluation.yml`,
`configs/phase_7_protocol.yml`) before out-of-sample scores are
computed; auctions are never shuffled across train/test.

## Local dashboard

A static, dependency-free dashboard (no external scripts, fonts, or
analytics) compares the point-forecast models at both cutoffs, interval
coverage alongside width, and the stress classifier against its
no-skill baseline — read from the small committed JSON aggregate, so it
needs no local data or live API calls.

```bash
python3 -m http.server 8000   # from the REPOSITORY ROOT, not dashboard/
# then open http://localhost:8000/dashboard/
```

## Testing and reproducibility

```bash
uv python install 3.12   # if needed; Python 3.12 is pinned
uv sync                  # create the environment from uv.lock
uv run pytest            # offline test suite (no network required)
uv run ruff check .      # lint
uv run python -m treasury_auction_stress.governance.audit_cli   # release audits
```

Reproducing the pipeline end to end (live network required; each stage
writes only to gitignored `data/` and `artifacts/` directories):

```bash
uv run python -m treasury_auction_stress.data.cli                       # auction ingestion
uv run python -m treasury_auction_stress.features.feature_matrix_cli    # point-in-time feature matrix
uv run python -m treasury_auction_stress.evaluation.phase6_cli          # baselines + chronological backtest
uv run python -m treasury_auction_stress.evaluation.phase7_cli          # adaptive/probabilistic/stress models
uv run python -m treasury_auction_stress.evaluation.phase8_cli          # interpretation + dashboard data
```

Generated reports and figures are intentionally untracked: every
generation CLI writes to the gitignored `artifacts/` directory by
default (created on demand; `--reports-dir` overrides), with
fail-before-write, atomic output behavior. See
`docs/reproducibility.md` for clean-environment verification,
`configs/release_manifest.yml` for the machine-readable tracked-file
manifest and accepted-artifact checksums, and
`src/treasury_auction_stress/governance/` for the release-audit and
history-audit tooling that gates publication.

## Limitations

- Results are retrospective; no prospective validation has been run
  yet. Prospective performance may differ.
- The target is an allocation-share proxy, not a direct balance-sheet
  or funding-stress measure.
- Primary-dealer statistics coverage begins 2013-04-03 (a real gap for
  2010–2013 auctions); some position buckets exist only from 2022.
- The stress-event classifier is not usable as a stress detector
  (worse-than-no-skill calibration, disclosed above).
- Announcement dates carry no verified intraday timestamp; the
  pipeline uses conservative date-level availability rules instead.

## License

This project's original code and original documentation are licensed
under the MIT License — see `LICENSE`. **The MIT License applies only
to this project's own original work.** It does not relicense, and does
not grant any right to, third-party data or source material
redistributed or referenced here — including U.S. Department of the
Treasury, Federal Reserve Bank of New York, Commodity Futures Trading
Commission, or Federal Reserve Bank of Philadelphia content — each of
which remains governed by its own publisher's terms (see
`docs/data_sources_and_licensing.md`), nor any trademark, logo, or
name of those institutions. Nothing in this repository states or
implies endorsement by any of them.
