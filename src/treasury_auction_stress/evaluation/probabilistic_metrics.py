"""Phase 7: probabilistic-forecast metrics -- pinball loss, empirical
interval coverage, and interval width. Reuses
`treasury_auction_stress.evaluation.metrics`'s unit convention (share
targets reported in percentage points) so probabilistic and point
metrics are never accidentally mixed in different units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasury_auction_stress.evaluation.metrics import unit_label

# (interval_label, lower_quantile, upper_quantile) -- must match
# configs/phase_7_protocol.yml's probabilistic.nominal_intervals_reported.
NOMINAL_INTERVALS: tuple[tuple[str, float, float], ...] = (
    ("50%", 0.25, 0.75),
    ("80%", 0.10, 0.90),
    ("90%", 0.05, 0.95),
)


def quantile_column_name(level: float) -> str:
    """The one, shared column-naming convention for a quantile-forecast
    column (e.g. `0.10` -> `"q0.10"`, never Python's default `str(0.1)`
    -> `"0.1"`, which would silently collide `0.10` and `0.1` and drop
    the trailing zero every other module here expects). Used by
    `treasury_auction_stress.models.gbm.fit_predict_gbm_quantiles` (the
    producer) and every function below (the consumer) so the two can
    never drift out of sync.
    """
    return f"q{level:.2f}"


def pinball_loss(actual: pd.Series, forecast: pd.Series, quantile: float) -> pd.Series:
    """Row-wise pinball (quantile) loss: `max(q*(y-f), (q-1)*(y-f))`, in
    the target's OWN raw units (scaling to percentage points happens in
    the summary functions below, exactly like
    `evaluation.metrics.compute_error_metrics`).
    """
    diff = actual - forecast
    return np.maximum(quantile * diff, (quantile - 1) * diff)


def _scale(target_name: str) -> float:
    return 100.0 if unit_label(target_name) == "percentage points" else 1.0


def compute_pinball_table(
    df: pd.DataFrame, *, quantile_levels: tuple[float, ...], target_name: str, actual_col: str = "actual"
) -> dict:
    """Mean pinball loss per quantile level (column `f"q{level}"` must
    exist), plus the mean across all levels -- both in the target's
    scaled (percentage-point, where applicable) units. Returns `n=0`
    metrics for an empty `df`, never raises or divides by zero.
    """
    scale = _scale(target_name)
    if df.empty:
        per_level = {f"pinball_{quantile_column_name(level)}": None for level in quantile_levels}
        return {"n": 0, "pinball_mean": None, "unit": unit_label(target_name), **per_level}

    per_level = {}
    losses = []
    for level in quantile_levels:
        col = quantile_column_name(level)
        loss = pinball_loss(df[actual_col], df[col], level) * scale
        per_level[f"pinball_{col}"] = float(loss.mean())
        losses.append(loss)
    pooled = pd.concat(losses, axis=0)
    return {"n": len(df), "pinball_mean": float(pooled.mean()), "unit": unit_label(target_name), **per_level}


def compute_coverage_table(
    df: pd.DataFrame, *, target_name: str, actual_col: str = "actual", intervals: tuple = NOMINAL_INTERVALS
) -> dict:
    """Empirical coverage and mean width for each nominal interval in
    `intervals`. `n=0` metrics (all `None`) for an empty `df`.
    """
    scale = _scale(target_name)
    if df.empty:
        out: dict = {"n": 0, "unit": unit_label(target_name)}
        for label, _low_q, _high_q in intervals:
            out[f"coverage_{label}"] = None
            out[f"width_{label}"] = None
        return out

    out = {"n": len(df), "unit": unit_label(target_name)}
    for label, low_q, high_q in intervals:
        lower = df[quantile_column_name(low_q)]
        upper = df[quantile_column_name(high_q)]
        covered = (df[actual_col] >= lower) & (df[actual_col] <= upper)
        out[f"coverage_{label}"] = float(covered.mean())
        out[f"width_{label}"] = float(((upper - lower) * scale).mean())
    return out


def summarize_probabilistic_by(
    df: pd.DataFrame,
    *,
    quantile_levels: tuple[float, ...],
    target_name: str,
    group_col: str,
    intervals: tuple = NOMINAL_INTERVALS,
) -> pd.DataFrame:
    """One row per distinct `group_col` value (e.g. `test_year`,
    `tenor`), with that group's own row count, pinball loss, and
    interval coverage/width -- so a report table always shows counts
    alongside a coverage fraction, never a bare percentage.
    """
    rows = []
    for value, group in df.groupby(group_col):
        pinball = compute_pinball_table(group, quantile_levels=quantile_levels, target_name=target_name)
        coverage = compute_coverage_table(group, target_name=target_name, intervals=intervals)
        rows.append({group_col: value, **pinball, **{k: v for k, v in coverage.items() if k not in ("n", "unit")}})
    return pd.DataFrame(rows)
