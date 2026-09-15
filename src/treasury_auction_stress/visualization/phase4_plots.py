"""Phase 4 diagnostic plots: one figure per market/macro source,
covering rate levels, CFTC positioning around disruption windows, and
an RTDSM real-revision demonstration. Each figure carries a source
caption; no causal language.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from treasury_auction_stress.data.cftc_release_calendar import DISRUPTION_WINDOW_RANGES


def _finish(fig: plt.Figure, path: Path, caption: str) -> Path:
    fig.text(0.01, 0.01, caption, fontsize=7, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_treasury_rate_levels(wide_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for maturity in ("2 Yr", "10 Yr", "30 Yr"):
        if maturity in wide_df.columns:
            ax.plot(wide_df["rate_date"], wide_df[maturity], label=maturity, linewidth=1)
    ax.set_title("Treasury daily par yield curve rates, 2 Yr / 10 Yr / 30 Yr")
    ax.set_xlabel("Date")
    ax.set_ylabel("Par yield (percent)")
    ax.legend()
    return _finish(
        fig, path,
        "Source: U.S. Treasury daily par yield curve (home.treasury.gov). "
        "Not obtained via FRED/ALFRED -- see docs/data_source_governance.md.",
    )


def plot_cftc_dealer_net_positioning(long_df: pd.DataFrame, contract_code: str, contract_name: str, path: Path) -> Path:
    sub = long_df.loc[long_df["contract_code"] == contract_code].sort_values("report_date")
    net = sub["dealer_positions_long_all"] - sub["dealer_positions_short_all"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub["report_date"], net, linewidth=1, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.5)
    for _name, start, end in DISRUPTION_WINDOW_RANGES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="tab:red", alpha=0.15)
    ax.set_title(f"CFTC Dealer/Intermediary net position, {contract_name} ({contract_code})")
    ax.set_xlabel("Report (observation) date")
    ax.set_ylabel("Net contracts (long minus short; spreading excluded)")
    return _finish(
        fig, path,
        "Source: CFTC TFF Futures Only (publicreporting.cftc.gov), dataset gpe5-46if. "
        "Shaded bands: the 3 documented release-calendar disruption windows "
        "(2018-19 and 2025 shutdowns, 2023 ION incident). Not FRED/ALFRED.",
    )


def plot_rtdsm_vintage_revision(long_df: pd.DataFrame, early_vintage: str, later_vintage: str, path: Path) -> Path:
    early = long_df.loc[long_df["vintage_label"] == early_vintage].sort_values("observation_date")
    later = long_df.loc[long_df["vintage_label"] == later_vintage].sort_values("observation_date")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(early["observation_date"], early["value"], label=f"vintage {early_vintage} (first known)", linewidth=1.4)
    ax.plot(later["observation_date"], later["value"], label=f"vintage {later_vintage} (later revision)", linewidth=1.4, linestyle="--")
    ax.set_title("RUC (unemployment rate): same observations, two real vintages")
    ax.set_xlabel("Observation month")
    ax.set_ylabel("Percent, seasonally adjusted")
    ax.legend()
    return _finish(
        fig, path,
        "Source: Federal Reserve Bank of Philadelphia RTDSM (philadelphiafed.org), RUC. "
        "Demonstrates real, non-fabricated data revision across vintages. Not FRED/ALFRED.",
    )
