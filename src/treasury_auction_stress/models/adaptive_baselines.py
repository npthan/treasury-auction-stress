"""Phase 7: the adaptive recent-history benchmark
(`configs/phase_7_protocol.yml`'s `adaptive_recent_history`).

Unlike Phase 6's `RecentHistoryBaseline` (frozen for the whole test
year at `fit` time), this model is allowed to use an already-resolved
EARLIER auction from within its own test year, but only once that
auction's own result satisfies the project's conservative
result-safe-availability rule relative to the auction currently being
predicted. This is implemented entirely via
`treasury_auction_stress.features.safe_as_of_lookback.
safe_as_of_trailing_mean` -- the same primitive the stress-event gate's
safe regime feature uses -- so there is exactly one tested
implementation of "safe trailing mean," not two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from treasury_auction_stress.evaluation.timing import RESULT_SAFE_AVAILABLE_DATE_COL
from treasury_auction_stress.features.safe_as_of_lookback import (
    safe_as_of_trailing_mean,
)

TENOR_COL = "tenor"
DATE_COL = "auction_date"
AUCTION_KEY_COL = "auction_key"

DEFAULT_N_LOOKBACK = 8


@dataclass
class AdaptiveRecentHistoryBaseline:
    """`fit(train_df)` records the frozen training pool and its global
    mean (used only as the last-resort fallback for a test row with
    zero eligible candidates of any kind). `predict(test_df,
    cutoff_col=...)` recomputes, for every test row, the safe trailing
    mean over the candidate pool UNION `test_df` using that row's own
    cutoff -- so an early-test-year auction's own already-resolved
    result can inform a later-test-year same-tenor prediction, exactly
    the "adaptive" behavior this model exists to test.

    ## Acceptance-review addition: `candidate_pool` (a documented, fixed narrower-definition edge case)

    `train_df` (the outer fold's own `fit_origin`-limited training
    pool, per `treasury_auction_stress.evaluation.timing.build_fold`)
    excludes any prior-year row whose own `result_safe_available_date`
    falls AFTER the fold's `fit_origin` -- correct for the FROZEN
    baseline, which must reflect only what was knowable at the start of
    the test year. But if `train_df` were also used as the ADAPTIVE
    model's own candidate-lookup pool, a prior-year "straggler" row
    excluded from `train_df` for that reason would be permanently
    invisible to the adaptive model for the entire test year, even
    after it genuinely becomes available -- WRONGLY narrower than the
    model's own stated purpose, since `safe_as_of_trailing_mean`'s
    per-row cutoff check (not `fit_origin`) is what actually decides
    eligibility. `candidate_pool` (optional; defaults to `train_df`
    when omitted, preserving prior behavior exactly) lets a caller pass
    a WIDER pool -- typically every row with `auction_date` in a year
    strictly before the test year, regardless of `fit_origin` -- so a
    straggler becomes usable the moment a LATER test-year row's own
    cutoff clears it, never earlier. `global_mean_` (the fallback) is
    still computed from `train_df` alone, never the wider pool -- using
    the wider pool for a single reused fallback CONSTANT would not be
    per-row-cutoff-checked and could leak a straggler into an
    early-test-year row's fallback. This scenario is verified to affect
    ZERO rows of this project's real 2010-2026 sample (every fold's own
    `n_train_dropped_for_year_boundary` is 0 -- see
    the Phase 7 acceptance review); the fix is disclosed and
    tested (`tests/test_phase7_adaptive_baseline.py::
    test_candidate_pool_lets_a_year_boundary_straggler_become_usable_once_available`)
    rather than left as an unexamined gap.
    """

    target_col: str = "primary_dealer_share"
    n_lookback: int = DEFAULT_N_LOOKBACK
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    candidate_pool_: pd.DataFrame = field(default=None, init=False, repr=False)
    n_train_: int = field(default=0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame, *, candidate_pool: pd.DataFrame | None = None) -> AdaptiveRecentHistoryBaseline:
        clean = train_df.dropna(subset=[self.target_col])
        if clean.empty:
            raise ValueError("AdaptiveRecentHistoryBaseline.fit: no non-missing training targets")
        self.global_mean_ = float(clean[self.target_col].mean())
        self.candidate_pool_ = candidate_pool.copy() if candidate_pool is not None else train_df.copy()
        self.n_train_ = len(clean)
        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame, *, cutoff_col: str) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("AdaptiveRecentHistoryBaseline.predict called before fit")
        if RESULT_SAFE_AVAILABLE_DATE_COL not in self.candidate_pool_.columns or RESULT_SAFE_AVAILABLE_DATE_COL not in test_df.columns:
            raise KeyError(
                f"AdaptiveRecentHistoryBaseline.predict: both the candidate pool and test frame must already carry "
                f"{RESULT_SAFE_AVAILABLE_DATE_COL!r} (via evaluation.timing.add_result_safe_available_date)"
            )

        # `ignore_index=True` plus a lookup keyed on `auction_key` (never
        # pandas positional index) avoids any dependence on -- or
        # collision between -- candidate_pool_'s and test_df's own index
        # values, which are not guaranteed disjoint in general callers.
        combined = pd.concat([self.candidate_pool_, test_df], ignore_index=True)
        result = safe_as_of_trailing_mean(
            combined,
            group_col=TENOR_COL,
            value_col=self.target_col,
            key_col=AUCTION_KEY_COL,
            cutoff_col=cutoff_col,
            safe_date_col=RESULT_SAFE_AVAILABLE_DATE_COL,
            date_col=DATE_COL,
            n_lookback=self.n_lookback,
        )
        mean_col = f"{self.target_col}_safe_trailing_mean"
        if result[AUCTION_KEY_COL].duplicated().any():
            raise ValueError("AdaptiveRecentHistoryBaseline.predict: auction_key is not unique across train+test")
        result_by_key = result.set_index(AUCTION_KEY_COL)[mean_col]
        forecast = test_df[AUCTION_KEY_COL].map(result_by_key).fillna(self.global_mean_)
        return pd.Series(forecast.to_numpy(), index=test_df.index)
