"""Phase 6 diagnostic figures. Both figures use ONLY out-of-sample
prediction rows (`data/processed/phase_6_oos_predictions.parquet`) --
never an in-sample fitted value -- and both clearly label the target
unit, cutoff view, and the provisional partial year.

Each `plot_*` function returns PNG bytes rather than writing to disk
itself -- `phase6_cli._generate` holds these in memory until its
mandatory gate test suite passes, and only then writes them atomically
(temp file + rename), so a failing gate can never leave a partially- or
freshly-written figure behind. Pass the returned bytes to
`write_figure_atomically` (or write them directly) to publish one.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def write_figure_atomically(png_bytes: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(png_bytes)
    tmp.replace(path)
    return path


def _finish(fig: plt.Figure, caption: str) -> bytes:
    fig.text(0.01, 0.01, caption, fontsize=7, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return buffer.getvalue()


def plot_annual_mae_by_method(
    predictions: pd.DataFrame, *, target_name: str, cutoff_view: str, model_ids: list[str], model_labels: dict[str, str]
) -> bytes:
    """Annual MAE (percentage points) by method, complete years as
    solid markers and the provisional partial year as a distinct,
    clearly-labeled hollow marker -- never blended visually with the
    complete-year line.
    """
    subset = predictions.loc[
        (predictions["target_name"] == target_name) & (predictions["cutoff_view"] == cutoff_view) & (predictions["model_id"].isin(model_ids))
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model_id in model_ids:
        model_subset = subset.loc[subset["model_id"] == model_id]
        complete = model_subset.loc[~model_subset["is_provisional"]]
        by_year = complete.groupby("test_year").apply(
            lambda g: (g["forecast"] - g["actual"]).abs().mean() * 100, include_groups=False
        )
        line = ax.plot(by_year.index, by_year.to_numpy(), marker="o", label=model_labels.get(model_id, model_id), linewidth=1.5)

        provisional = model_subset.loc[model_subset["is_provisional"]]
        if not provisional.empty:
            prov_mae = (provisional["forecast"] - provisional["actual"]).abs().mean() * 100
            ax.plot(
                provisional["test_year"].iloc[0],
                prov_mae,
                marker="D",
                markerfacecolor="none",
                markeredgewidth=1.5,
                color=line[0].get_color(),
            )

    ax.set_title(f"Annual out-of-sample MAE by method -- {target_name} ({cutoff_view} cutoff)")
    ax.set_xlabel("Test year (hollow diamond = provisional partial year)")
    ax.set_ylabel("MAE (percentage points)")
    ax.legend(fontsize=8, ncol=2)
    caption = (
        f"Out-of-sample only. Target: {target_name} (percentage points). Cutoff view: {cutoff_view}. "
        "Complete calendar-year folds shown as solid circles/lines; the current, incomplete calendar year "
        "is shown as a separate hollow diamond and is never averaged into a complete-year statistic."
    )
    return _finish(fig, caption)


def plot_actual_vs_forecast_over_time(
    predictions: pd.DataFrame, *, target_name: str, cutoff_view: str, model_id: str, model_label: str
) -> bytes:
    """Out-of-sample actual vs. forecast over time for one model, with
    the provisional partial year shaded distinctly.
    """
    subset = predictions.loc[
        (predictions["target_name"] == target_name)
        & (predictions["cutoff_view"] == cutoff_view)
        & (predictions["model_id"] == model_id)
    ].sort_values("test_prediction_cutoff")

    scale = 100.0 if target_name in {"primary_dealer_share", "direct_bidder_share", "indirect_bidder_share"} else 1.0
    unit = "percentage points" if scale == 100.0 else "ratio units"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    dates = pd.to_datetime(subset["test_prediction_cutoff"])
    ax.plot(dates, subset["actual"] * scale, label="Actual", linewidth=1, color="black")
    ax.plot(dates, subset["forecast"] * scale, label="Out-of-sample forecast", linewidth=1, alpha=0.8)

    provisional_start = subset.loc[subset["is_provisional"], "test_prediction_cutoff"]
    if not provisional_start.empty:
        ax.axvspan(pd.to_datetime(provisional_start.min()), dates.max(), color="grey", alpha=0.15, label="Provisional partial year")

    ax.set_title(f"Out-of-sample actual vs. forecast -- {model_label}, {target_name} ({cutoff_view} cutoff)")
    ax.set_xlabel("Test prediction cutoff date")
    ax.set_ylabel(f"{target_name} ({unit})")
    ax.legend(fontsize=8)
    caption = (
        f"Out-of-sample only (each point is that auction's frozen annual-fold forecast). "
        f"Target unit: {unit}. Cutoff view: {cutoff_view}. Model: {model_label}. "
        "Shaded region is the provisional, incomplete current calendar year."
    )
    return _finish(fig, caption)
