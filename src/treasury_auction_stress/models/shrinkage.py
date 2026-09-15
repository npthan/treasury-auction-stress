"""Phase 7: `challenger_shrinkage_tenor_reopening`
(`configs/phase_7_protocol.yml`'s `shrinkage_challenger`) -- an
empirical Bayes / James-Stein-style shrinkage estimator, explicitly NOT
a full Bayesian model.

Two-level shrinkage, fit strictly inside one fold's own training pool:

1. Each TENOR's own mean is shrunk toward the training pool's global
   mean.
2. Each (tenor, is_reopening) cell's own mean is shrunk toward its OWN
   tenor's already-shrunk level-1 mean -- the "wider historical pool" a
   sparse cell borrows strength from.

At each level, the shrinkage weight on a group's own (noisy) mean is
`w_g = n_g / (n_g + k)`, where `k = within_group_variance /
between_group_variance`, both estimated ONCE per fold via the standard
one-way random-effects ANOVA method-of-moments estimator (see
`_anova_variance_components` below) -- a data-driven constant, not a
hyperparameter searched via inner cross-validation.

This differs from Phase 6's `TenorReopeningMeanBaseline` (an unshrunk
group mean with an all-or-nothing fallback hierarchy for a genuinely
EMPTY group) by giving every group, even one with a HANDFUL of
observations, continuously-graded partial credit toward a wider pool --
`TenorReopeningMeanBaseline`'s fallback only ever engages for a group
with literally zero training rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TENOR_COL = "tenor"
REOPENING_COL = "is_reopening"

MIN_VARIANCE_FLOOR = 1e-8


def _anova_variance_components(df: pd.DataFrame, group_col, value_col: str) -> tuple[float, float, dict, dict, float]:
    """Standard one-way (possibly unbalanced) random-effects ANOVA
    method-of-moments estimator. Returns
    `(within_group_variance, between_group_variance, group_means, group_ns, grand_mean)`.

    `between_group_variance` is floored at `MIN_VARIANCE_FLOOR` (never
    zero or negative, which the method-of-moments estimator can
    otherwise produce for a small or genuinely homogeneous sample) --
    this means a fold with near-identical group means shrinks nearly
    all the way to the wider pool, never producing a division by zero.
    """
    grouped = df.groupby(group_col)[value_col]
    group_means = grouped.mean().to_dict()
    group_ns = grouped.count().to_dict()
    group_vars = grouped.var(ddof=1)  # NaN for any singleton group

    n_total = int(sum(group_ns.values()))
    k_groups = len(group_ns)
    grand_mean = float(sum(group_means[g] * group_ns[g] for g in group_means) / n_total)

    if k_groups < 2:
        # Only one group exists -- nothing to shrink toward besides
        # itself; between-group variance is not estimable, and the
        # caller's own w_g=n/(n+k) formula is never invoked in this
        # case (see ShrinkageTenorReopeningBaseline.fit).
        return 0.0, MIN_VARIANCE_FLOOR, group_means, group_ns, grand_mean

    ssb = sum(group_ns[g] * (group_means[g] - grand_mean) ** 2 for g in group_means)
    msb = ssb / (k_groups - 1)

    ssw = float(sum((group_ns[g] - 1) * group_vars.get(g, 0.0) for g in group_means if group_ns[g] > 1))
    df_within = n_total - k_groups
    if df_within > 0:
        msw = ssw / df_within
    else:
        # Degenerate case: every group has exactly 1 observation, so
        # within-group noise cannot be estimated from repeated draws
        # within a group at all. Fall back to the overall sample
        # variance of `value_col` as a disclosed, conservative proxy
        # for within-group noise (never left undefined / division by
        # zero).
        msw = float(df[value_col].var(ddof=1)) if len(df) > 1 else 0.0

    sum_n_sq = sum(n**2 for n in group_ns.values())
    n0 = (n_total - sum_n_sq / n_total) / (k_groups - 1)
    between_variance = (msb - msw) / n0 if n0 > 0 else 0.0
    between_variance = max(between_variance, MIN_VARIANCE_FLOOR)

    return float(msw), float(between_variance), group_means, group_ns, grand_mean


@dataclass
class ShrinkageTenorReopeningBaseline:
    target_col: str = "primary_dealer_share"
    is_fitted_: bool = field(default=False, init=False, repr=False)
    global_mean_: float = field(default=float("nan"), init=False, repr=False)
    tenor_shrunk_: dict = field(default_factory=dict, init=False, repr=False)
    tenor_weight_: dict = field(default_factory=dict, init=False, repr=False)
    tenor_n_: dict = field(default_factory=dict, init=False, repr=False)
    k_tenor_: float = field(default=0.0, init=False, repr=False)
    group_shrunk_: dict = field(default_factory=dict, init=False, repr=False)
    group_weight_: dict = field(default_factory=dict, init=False, repr=False)
    group_n_: dict = field(default_factory=dict, init=False, repr=False)
    k_group_: float = field(default=0.0, init=False, repr=False)

    def fit(self, train_df: pd.DataFrame) -> ShrinkageTenorReopeningBaseline:
        clean = train_df.dropna(subset=[self.target_col]).copy()
        if clean.empty:
            raise ValueError("ShrinkageTenorReopeningBaseline.fit: no non-missing training targets")

        self.global_mean_ = float(clean[self.target_col].mean())

        # Level 1: tenor means shrunk toward the global mean.
        within_t, between_t, tenor_means, tenor_ns, _grand_mean_t = _anova_variance_components(
            clean, TENOR_COL, self.target_col
        )
        self.k_tenor_ = within_t / between_t if len(tenor_means) >= 2 else 0.0
        for tenor, mean_ in tenor_means.items():
            n = tenor_ns[tenor]
            weight = 1.0 if len(tenor_means) < 2 else n / (n + self.k_tenor_)
            self.tenor_weight_[tenor] = weight
            self.tenor_shrunk_[tenor] = weight * mean_ + (1.0 - weight) * self.global_mean_
            self.tenor_n_[tenor] = n

        # Level 2: (tenor, is_reopening) cell means shrunk toward their
        # OWN tenor's already-shrunk level-1 mean.
        group_key = pd.Series(
            list(zip(clean[TENOR_COL], clean[REOPENING_COL], strict=True)), index=clean.index
        )
        clean = clean.assign(_group_key=group_key)
        within_g, between_g, group_means, group_ns, _grand_mean_g = _anova_variance_components(
            clean, "_group_key", self.target_col
        )
        self.k_group_ = within_g / between_g if len(group_means) >= 2 else 0.0
        for key, mean_ in group_means.items():
            tenor = key[0]
            n = group_ns[key]
            prior = self.tenor_shrunk_.get(tenor, self.global_mean_)
            weight = 1.0 if len(group_means) < 2 else n / (n + self.k_group_)
            self.group_weight_[key] = weight
            self.group_shrunk_[key] = weight * mean_ + (1.0 - weight) * prior
            self.group_n_[key] = n

        self.is_fitted_ = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("ShrinkageTenorReopeningBaseline.predict called before fit")

        def _one(row: pd.Series) -> float:
            key = (row[TENOR_COL], bool(row[REOPENING_COL]))
            if key in self.group_shrunk_:
                return self.group_shrunk_[key]
            if row[TENOR_COL] in self.tenor_shrunk_:
                return self.tenor_shrunk_[row[TENOR_COL]]
            return self.global_mean_

        return test_df.apply(_one, axis=1)

    def describe_shrinkage(self) -> pd.DataFrame:
        """A small, reportable table: one row per (tenor, is_reopening)
        cell, its own raw mean is not stored directly here (see
        `group_shrunk_`'s docstring context) but its weight, n, shrunk
        value, and the tenor-level prior it was shrunk toward are --
        exactly the "own mean / wider-pool mean / shrinkage weight"
        triple this model's interpretability rests on.
        """
        if not self.is_fitted_:
            raise RuntimeError("ShrinkageTenorReopeningBaseline.describe_shrinkage called before fit")
        rows = []
        for (tenor, is_reopening), shrunk in self.group_shrunk_.items():
            rows.append(
                {
                    "tenor": tenor,
                    "is_reopening": is_reopening,
                    "n": self.group_n_[(tenor, is_reopening)],
                    "shrinkage_weight_on_own_mean": self.group_weight_[(tenor, is_reopening)],
                    "wider_pool_prior_tenor_level": self.tenor_shrunk_.get(tenor, self.global_mean_),
                    "shrunk_estimate": shrunk,
                }
            )
        return pd.DataFrame(rows).sort_values(["tenor", "is_reopening"]).reset_index(drop=True)
