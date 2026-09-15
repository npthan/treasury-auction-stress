# Target Specification

**Status: finalized for Phase 2, accepted after review.** The formulas
below were verified against the live, normalized Treasury Fiscal Data
auctions table (1281 nominal-coupon rows, 2010-01-01 through this
session's retrieval date; see `artifacts/auction_data_quality.md` and
`artifacts/auction_data_profile.md`, both regenerated from code). They
supersede the Phase 0/1 provisional draft. Where this document
disagrees with an earlier phase's assumption, this document is
correct and the earlier one was wrong — that earlier draft explicitly
said as much ("the verified one wins").

**This document was revised once already, by a Phase 2 acceptance
review** that found and fixed a real formula error (FIMA add-ons had
been wrongly excluded from the share denominator; see the section
below) and a real classification gap (two auctions needed a verified,
documented special-auction exclusion, not just a flag). See
the Phase 2 acceptance review for the full investigation,
primary-source citations, and before/after numbers.

Implementation lives in `src/treasury_auction_stress/features/`:
`targets.py` (the four candidate outcomes and bid-to-cover
reconciliation), `eligibility.py` (sample selection), and
`dealer_absorption.py` (the leakage-safe Dealer Absorption Surprise
transformer and stress-label threshold). Every claim below is backed
by a test in `tests/test_targets.py`, `tests/test_eligibility.py`, or
`tests/test_dealer_absorption.py`.

## The key finding this phase made: `total_accepted` is not what it looks like

The provisional spec assumed a share's denominator would simply be
"total accepted amount." Verified by exact-dollar reconciliation over
the full 1281-row settled sample (zero exceptions):

```
total_accepted == comp_accepted + noncomp_accepted
                   + fima_noncomp_accepted + soma_accepted

comp_accepted   == primary_dealer_accepted + direct_bidder_accepted
                    + indirect_bidder_accepted
```

`soma_accepted` (Federal Reserve System Open Market Account add-ons —
the Fed rolling its own maturing Treasury holdings into the new issue)
is awarded **outside competitive bidding**, on top of the publicly
announced offering amount, and represents the Fed's own portfolio
operation, not any market participant's demand.

`fima_noncomp_accepted` (Foreign and International Monetary
Authorities noncompetitive add-ons) is a **different kind of
quantity**. The acceptance review read Treasury's own auction-results
press releases directly (not just the API) and found that Treasury's
own published "Subtotal" line — the basis for its own published
Bid-to-Cover Ratio — is `Competitive + Noncompetitive + FIMA
(Noncompetitive)`, explicitly **excluding only SOMA**. FIMA
noncompetitive bids are real, allocated demand from a real bidder
category (foreign official institutions bidding directly, guaranteed
fill, analogous to domestic noncompetitive bids), unlike SOMA, which
is the Fed's own portfolio rollover. An earlier version of this
document (and of `treasury_auction_stress.features.targets`) excluded
FIMA as well as SOMA; that was a genuine formula error, not a rounding
artifact, and the acceptance review's bid-to-cover reconciliation is
what caught it — see the Phase 2 acceptance review.

Across the settled sample, the SOMA add-on alone averages roughly
6-8% of `total_accepted` per tenor but exceeds 40% for some individual
auctions (concentrated, though not exclusively, in 2020-2022). Using
`total_accepted` (which includes SOMA) as a share denominator would
materially *understate* every bidder category's true share of the
auction's actually-competed portion — by as much as 13 percentage
points for some individual auctions (see
`artifacts/auction_data_profile.md`, "Extreme observations").

**This project uses `public_accepted_amount = comp_accepted +
noncomp_accepted + fima_noncomp_accepted` as the denominator for every
bidder-category share and for bid-to-cover — matching Treasury's own
"Subtotal" exactly.** This amount is verified to partition *exactly*
into five categories (dealer, direct, indirect, noncompetitive, FIMA),
to the dollar, over the entire settled sample.

## The four candidate outcomes (finalized)

All four share the same denominator, `public_accepted_amount`. Values
are missing (`pd.NA`) whenever the auction hasn't settled yet — never
zero, never imputed. All four are computed on the **main modeling
sample** (`treasury_auction_stress.features.eligibility.select_modeling_sample`),
which excludes the two verified special auctions discussed below.

1. **Primary-dealer take-down / allotment share** (central quantity of interest)
   ```
   primary_dealer_share = primary_dealer_accepted / public_accepted_amount
   ```
   Verified: `primary_dealer_accepted` is never exactly zero anywhere
   in the 2010-2026 settled sample — consistent with primary dealers'
   obligation to bid at every auction. `direct_bidder_accepted` and
   `indirect_bidder_accepted` are each exactly zero for the same 2
   auctions (the two special, restricted auctions excluded from the
   modeling sample — see below); these are genuine zeros, mechanically
   required by Treasury's own bidding restriction on those auctions,
   distinguishable from missingness via `results_available`.

2. **Direct-bidder share**
   ```
   direct_bidder_share = direct_bidder_accepted / public_accepted_amount
   ```

3. **Indirect-bidder share**
   ```
   indirect_bidder_share = indirect_bidder_accepted / public_accepted_amount
   ```

4. **Bid-to-cover ratio** — the API already provides this; see below.

Two more diagnostic-only quantities (`noncompetitive_share =
noncomp_accepted / public_accepted_amount` and `fima_share =
fima_noncomp_accepted / public_accepted_amount`) are computed alongside
the four so that `primary_dealer_share + direct_bidder_share +
indirect_bidder_share + noncompetitive_share + fima_share == 1.0`
exactly for every settled row — a built-in consistency check, verified
in `tests/test_targets.py`.

### Treatment of noncompetitive bids (domestic and FIMA)

Both domestic noncompetitive bids and FIMA noncompetitive bids are
**included in the denominator** (they are part of the amount actually
issued and allocated) but **never included in a primary bidder
category's numerator** — each is its own diagnostic category.

### Why `public_accepted_amount`, not `total_accepted`, is the economically appropriate denominator

The research question is "when other demand is weak, how much of the
issue do dealers have to warehouse." SOMA add-ons are not part of "the
issue" in that sense — they are the Fed's own portfolio operation
layered on top of it, with no relationship to a dealer's willingness
to absorb supply. Including SOMA in the denominator would let a large,
unrelated Fed rollover mechanically shrink every bidder category's
computed share, without any change in actual market demand — exactly
the kind of confound this document's "why high dealer take-down is not
automatically proof of a weak auction" discussion already warned about
for offering size in general. FIMA add-ons, by contrast, belong in the
denominator precisely because they *are* real market demand (from a
different bidder category), which is also exactly what Treasury's own
published methodology does.

## Bid-to-cover: the API already provides it, and it now reconciles exactly

`bid_to_cover_ratio` is a precomputed field, not something this
project needs to invent from scratch. This phase reconciled it against
a from-scratch calculation and found:

```
bid_to_cover_ratio == round(
    (comp_tendered + noncomp_accepted + fima_noncomp_accepted)
    / public_accepted_amount,
    2,
)
```

i.e. Treasury's own published ratio **excludes only SOMA** from both
numerator and denominator — a "market demand" ratio, not a "total
issuance including the Fed's rollover" ratio. Verified over the full
main-modeling-sample rows (n=1279, after excluding the two special
auctions): **100% match exactly** (max absolute difference 0.0000).
This is a correction from this phase's own earlier draft, which
reported "93.3% within 0.01, 99.8% within 0.02, small unexplained
residual" — that residual was not rounding noise, it was FIMA being
wrongly excluded; see the Phase 2 acceptance review for the
full reconciliation table and the primary-source proof (a worked
example quoted directly from a Treasury results press release: "Bid-
to-Cover Ratio: $85,850,000/$25,000,400 = 3.43"). This project keeps
the API's own `bid_to_cover_ratio` as the primary field and exposes
the from-scratch `bid_to_cover_calculated` purely as an audit
cross-check — any future discrepancy is now a genuine finding worth
investigating, not expected noise.

## Auction eligibility (finalized)

See `src/treasury_auction_stress/features/eligibility.py` for the
full, authoritative documentation (reproduced in summary here):

- **Nominal-coupon inclusion rule** (unchanged from Phase 1):
  `security_type in {"Note", "Bond"}` AND `inflation_index_security ==
  "No"` AND `floating_rate == "No"`.
- **Supported tenors**: all seven (2/3/5/7/10/20/30-year). The
  20-year bond has no history before its May 2020 reintroduction —
  a permanent structural limitation, not a gap to fill in.
- **New issues vs. reopenings**: both retained as separate
  observations, per `docs/project_plan.md`.
- **Completed vs. pending**: an auction with `results_available ==
  False` is split into a separate `pending` frame, never mixed into
  the settled analytical sample and never dropped silently.
- **Cancelled auctions**: no cancellation field exists anywhere in the
  verified live schema; none were found because none could be
  represented.
- **Records with incomplete result fields**: verified zero — a settled
  auction's result fields are either all present or (if pending) all
  missing; no partial-missingness case exists in the 2010-2026 sample.
- **Duplicates**: zero, on `(cusip, auction_date)`.
- **Revised records**: not yet detectable with a single retrieval
  snapshot per day; open question, tracked in
  `artifacts/auction_data_profile.md`.
- **Analysis start date**: `2010-01-01`, unchanged from Phase 1.
- **Special auction category (verified): restricted, primary-dealer-only
  reopenings.** Two settled auctions (2019-06-21, CUSIP `9128286T2`,
  10-year reopening; 2021-12-02, CUSIP `912810TC2`, 20-year reopening)
  were investigated directly against Treasury's own official
  announcement and results press releases (URLs and quoted text in
  `treasury_auction_stress.features.eligibility` and
  the Phase 2 acceptance review). Both announcements
  explicitly restrict bidding to primary dealers only, submitted by
  phone/email directly to the New York Fed — "Customer bids will not
  be accepted. Noncompetitive tenders, including FIMA tenders, will
  not be accepted." This is a **verified, resolved finding**, not a
  guess from the $25,000,000 offering size alone: their near-100%
  dealer shares are a mechanical, rule-imposed consequence, not
  evidence of auction stress. They are **preserved** in the complete
  normalized dataset and in the `settled` audit frame, tagged via
  `special_auction_type`, and **excluded** from the main modeling
  sample by `select_modeling_sample` — an explicit function, not a
  silent filter. `is_unusually_small_offering` (the dollar-threshold
  flag) is retained as a secondary, corroborating signal only.

## Dealer Absorption Surprise (implemented, leakage-safe)

Implemented in `treasury_auction_stress.features.dealer_absorption`.
Unchanged in concept from the provisional draft: the residual of
`primary_dealer_share` after removing the part predictable from
tenor, reopening status, offering size, and a slowly-changing
historical regime.

```
expected_share(auction) =
    group_mean[tenor, is_reopening]                            (learned on training fold)
  + offering_slope * (log(offering_amt) - offering_center)     (learned, 1 coefficient)
  + regime_slope   * (regime_feature - regime_center)          (learned, 1 coefficient)

dealer_absorption_surprise = primary_dealer_share - expected_share
```

- `regime_feature` (`dealer_share_regime_trailing_mean`) is the
  trailing mean of `primary_dealer_share` over up to 8 preceding
  same-tenor auctions, **excluding the current auction**
  (`shift(1)` before `rolling(...)`) — safe to compute once over the
  full eligible history because each row only ever depends on
  strictly-earlier rows in `auction_date` order. **This is a narrower
  claim than "safe to use as a real-time Phase 6 predictor"** —
  "strictly-earlier in `auction_date` order" is not automatically the
  same thing as "publicly, safely known by this auction's own forecast
  cutoff." Phase 6's acceptance review
  (`treasury_auction_stress.evaluation.dealer_absorption_audit`)
  checked this directly against real data and found real cases where
  the two diverge (up to 34 (auction, lookback-slot, cutoff)
  violations across 18 distinct auctions, checking the full 8-auction
  window at both prediction cutoffs) — see
  `artifacts/phase_6_evaluation_protocol.md` and `docs/point_in_time_rules.md`'s
  Phase 6 section for the current counts and the reasoning. This is why
  Phase 6 does not use this feature as a predictor and defers Dealer
  Absorption Surprise modeling to Phase 7.
- `DealerAbsorptionSurpriseModel.fit` learns `group_mean`,
  `offering_slope`, `regime_slope`, and the two centering constants
  from a training-fold dataframe only. `.transform` applies the
  frozen parameters to any dataframe — proven, in
  `tests/test_dealer_absorption.py`, to give bit-identical output for
  historical rows whether or not future rows are present in the input.
- Deliberately interpretable: group means plus two linear
  coefficients (fit by ordinary least squares), not a black-box model.
  A more sophisticated residualization model is explicit future-phase
  work.
- `artifacts/auction_data_profile.md` contains a full-history,
  walk-forward (expanding-origin, annual-fold) construction of this
  target for descriptive analysis — the first calendar year of history
  is excluded (no strictly-prior training fold exists yet). Its
  distribution has a **negative mean** (~-0.023), which the profile
  explains: primary-dealer share has trended steadily downward over
  2010-2026, and an expanding-origin model that only sees the past
  necessarily lags a persistent one-directional trend. This is an
  expected property of a simple training-only baseline, not a leakage
  bug — but it means the surprise's scale should be read fold-by-fold,
  not assumed to be zero-centered overall.

## Stress-event label (implemented, leakage-safe, provisional threshold)

Implemented as `StressThreshold` in the same module: `.fit(train_surprise)`
learns a percentile threshold from a training-fold surprise series
only; `.transform` flags `surprise >= threshold_` without ever
recomputing the threshold from its input (proven in
`tests/test_dealer_absorption.py` — passing extreme "test-period"
values into `.transform` never moves `threshold_`). Missing surprise
values transform to a missing flag, never `False`.

`describe_percentile_thresholds` compares several candidate
percentiles (75th/80th/90th/95th) descriptively, computed from a
training-fold series only. `artifacts/auction_data_profile.md` runs
this using an illustrative training window (surprise dated before
2020-01-01) purely for descriptive comparison in this phase — that
split is explicitly **not** a chronological evaluation fold and must
be re-derived, not reused, inside any real evaluation protocol in a
later phase. No threshold was chosen using any out-of-sample
performance metric.

## Why high dealer take-down is not automatically proof of a weak auction

Unchanged from the provisional draft — still the guiding principle for
interpreting every number above:

- **Mechanical backstop role.** Primary dealers must bid at every
  auction and are the residual buyer by design; some baseline
  take-down is structurally normal.
- **Distribution lag, not distress.** Dealers often intend to
  distribute securities to clients over the following hours/days; a
  high take-down at the moment of auction can reflect normal
  intermediation timing.
- **Confounded by offering size and, this phase discovered, the Fed's
  own rollover.** A larger offering, or a large SOMA rollover on top
  of it, can mechanically move measured shares without any change in
  underlying demand quality — which is exactly why this phase
  corrected the denominator (excluding SOMA, but correctly including
  FIMA as genuine third-party demand) and why tenor/size adjustment
  matters before calling anything a "surprise."
- **Regime dependence.** What counts as "normal" dealer participation
  has drifted over time (verified: mean primary-dealer share fell from
  ~38% pre-pandemic to ~14% after 2021, pooled across tenors — see
  `artifacts/auction_data_profile.md`). A share that would be alarming in
  one regime is unremarkable in another.

This project treats dealer take-down as a **continuous,
structurally-adjusted quantity to forecast and explain**, not a
pre-judged indicator of auction "success" or "failure."

## Optional, conditional target: auction tail / stop-through

Unchanged from the provisional draft and still not built: requires a
legitimate, point-in-time when-issued yield source, not yet
identified or authorized (see `configs/sources.yml`).
