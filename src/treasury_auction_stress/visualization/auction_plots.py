"""Phase 2 exploratory plots for the nominal-coupon auction sample.

Every function here takes an already-prepared dataframe (eligibility
applied, targets computed -- see
`treasury_auction_stress.features.profile_cli`) and a destination
path, draws exactly one figure, and returns the path it wrote. Each
figure carries a descriptive title, axis labels with units, a caption
naming the sample and data source, and no causal language -- these are
observational summaries of an auction dataset, not claims about what
caused what.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from treasury_auction_stress.data.schema import NOMINAL_COUPON_TENORS

SOURCE_CAPTION = (
    "Source: U.S. Treasury Fiscal Data, Treasury Securities Auctions API "
    "(fiscaldata.treasury.gov). Sample: nominal-coupon (2/3/5/7/10/20/30-year "
    "note/bond) auctions, 2010-01-01 onward, settled results only unless noted. "
    "Excludes 2 verified restricted, primary-dealer-only reopening auctions "
    "(see the Phase 2 acceptance review)."
)

COVID_ONSET = pd.Timestamp("2020-03-01")
COVID_END_APPROX = pd.Timestamp("2021-12-31")

_TENOR_ORDER = list(NOMINAL_COUPON_TENORS)


def _finish(fig: plt.Figure, path: Path, caption: str = SOURCE_CAPTION) -> Path:
    fig.text(0.01, 0.01, caption, fontsize=7, color="dimgray", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_auction_counts_by_year_tenor(settled: pd.DataFrame, path: Path) -> Path:
    counts = (
        settled.assign(year=settled["auction_date"].dt.year)
        .groupby(["year", "tenor"], observed=True)
        .size()
        .unstack("tenor")
        .reindex(columns=_TENOR_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Nominal-coupon Treasury auction counts by year and tenor")
    ax.set_xlabel("Auction year")
    ax.set_ylabel("Number of auctions")
    ax.legend(title="Tenor", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _finish(fig, path)


def plot_new_vs_reopening_by_tenor(settled: pd.DataFrame, path: Path) -> Path:
    counts = (
        settled.assign(status=settled["is_reopening"].map({True: "Reopening", False: "New issue"}))
        .groupby(["tenor", "status"], observed=True)
        .size()
        .unstack("status")
        .reindex(_TENOR_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.7, color=["#4C72B0", "#DD8452"])
    ax.set_title("New issues vs. reopenings by tenor, 2010-2026")
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Number of auctions")
    ax.legend(title=None, fontsize=8)
    return _finish(fig, path)


def plot_offering_amount_over_time(settled: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tenor in _TENOR_ORDER:
        sub = settled.loc[settled["tenor"] == tenor].sort_values("auction_date")
        if sub.empty:
            continue
        ax.plot(
            sub["auction_date"],
            sub["offering_amt"].astype("float64") / 1e9,
            marker=".",
            markersize=3,
            linewidth=0.8,
            label=tenor,
        )
    ax.set_title("Announced offering amount over time, by tenor")
    ax.set_xlabel("Auction date")
    ax.set_ylabel("Offering amount ($ billions)")
    ax.legend(title="Tenor", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _finish(fig, path)


def plot_bid_to_cover_over_time(settled: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tenor in _TENOR_ORDER:
        sub = settled.loc[settled["tenor"] == tenor].sort_values("auction_date")
        if sub.empty:
            continue
        ax.plot(
            sub["auction_date"],
            sub["bid_to_cover_ratio"].astype("float64"),
            marker=".",
            markersize=3,
            linewidth=0.8,
            label=tenor,
        )
    ax.set_title("Bid-to-cover ratio over time, by tenor (Treasury's published figure)")
    ax.set_xlabel("Auction date")
    ax.set_ylabel("Bid-to-cover ratio (total tendered / total accepted, unitless)")
    ax.legend(title="Tenor", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _finish(fig, path)


def plot_bidder_shares_by_tenor_small_multiples(settled: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, tenor in zip(axes, _TENOR_ORDER, strict=False):
        sub = settled.loc[settled["tenor"] == tenor].sort_values("auction_date")
        ax.plot(sub["auction_date"], sub["primary_dealer_share"].astype("float64"), label="Primary dealer", linewidth=0.9)
        ax.plot(sub["auction_date"], sub["direct_bidder_share"].astype("float64"), label="Direct", linewidth=0.9)
        ax.plot(sub["auction_date"], sub["indirect_bidder_share"].astype("float64"), label="Indirect", linewidth=0.9)
        ax.set_title(tenor, fontsize=10)
        ax.set_ylim(0, 1)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.06), fontsize=9)
    fig.suptitle(
        "Primary-dealer, direct, and indirect bidder shares over time, by tenor\n"
        "(share of public_accepted_amount = comp_accepted + noncomp_accepted + fima_noncomp_accepted; excludes only SOMA add-ons)",
        fontsize=11,
    )
    for ax in axes[-2:-1]:
        ax.set_xlabel("Auction date")
    return _finish(fig, path)


def plot_target_distribution_by_tenor(settled: pd.DataFrame, path: Path) -> Path:
    data = [
        settled.loc[settled["tenor"] == tenor, "primary_dealer_share"].dropna().astype("float64")
        for tenor in _TENOR_ORDER
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.boxplot(data, tick_labels=_TENOR_ORDER, showmeans=True)
    ax.set_title("Primary-dealer take-down share distribution by tenor")
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Primary-dealer share of public_accepted_amount (fraction, 0-1)")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, path)


def plot_target_distribution_by_reopening(settled: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    groups = []
    labels = []
    for tenor in _TENOR_ORDER:
        for is_reopening, label_suffix in ((False, "new"), (True, "reopen")):
            vals = settled.loc[
                (settled["tenor"] == tenor) & (settled["is_reopening"] == is_reopening),
                "primary_dealer_share",
            ].dropna().astype("float64")
            if vals.empty:
                continue
            groups.append(vals)
            labels.append(f"{tenor}\n{label_suffix}")
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_title("Primary-dealer share distribution by tenor and reopening status")
    ax.set_ylabel("Primary-dealer share of public_accepted_amount (fraction, 0-1)")
    ax.tick_params(axis="x", labelsize=7)
    return _finish(fig, path)


def plot_bidder_share_relationships(settled: pd.DataFrame, path: Path) -> Path:
    cols = ["primary_dealer_share", "direct_bidder_share", "indirect_bidder_share"]
    labels = ["Primary dealer", "Direct", "Indirect"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, (col_i, label_i) in enumerate(zip(cols, labels, strict=True)):
        for j, (col_j, label_j) in enumerate(zip(cols, labels, strict=True)):
            ax = axes[i, j]
            if i == j:
                ax.hist(settled[col_i].dropna().astype("float64"), bins=25, color="#4C72B0")
            else:
                ax.scatter(
                    settled[col_j].astype("float64"),
                    settled[col_i].astype("float64"),
                    s=4,
                    alpha=0.3,
                )
            if i == 2:
                ax.set_xlabel(label_j, fontsize=8)
            if j == 0:
                ax.set_ylabel(label_i, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Pairwise relationships among bidder-category shares (main modeling sample)")
    return _finish(fig, path)


def plot_dealer_share_structural_view(settled: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sub = settled.sort_values("auction_date")
    ax.scatter(
        sub["auction_date"],
        sub["primary_dealer_share"].astype("float64"),
        s=6,
        alpha=0.35,
        label="Individual auctions (all tenors pooled)",
    )
    pooled_rolling = (
        sub.set_index("auction_date")["primary_dealer_share"].astype("float64").rolling("365D").mean()
    )
    ax.plot(pooled_rolling.index, pooled_rolling.to_numpy(), color="black", linewidth=1.5, label="~1-year trailing mean")
    ax.axvspan(COVID_ONSET, COVID_END_APPROX, color="gray", alpha=0.15, label="2020-03 to 2021-12 (pandemic period)")
    ax.set_title(
        "Primary-dealer share over time, all tenors pooled\n"
        "(descriptive only -- no causal claim about any specific date or event)"
    )
    ax.set_xlabel("Auction date")
    ax.set_ylabel("Primary-dealer share of public_accepted_amount (fraction, 0-1)")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_dealer_absorption_surprise_distribution(walkforward_das: pd.DataFrame, path: Path) -> Path:
    values = walkforward_das["dealer_absorption_surprise"].dropna().astype("float64")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(values, bins=40, color="#55A868")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        "Dealer Absorption Surprise distribution\n"
        "(walk-forward, expanding-origin annual folds -- see artifacts/auction_data_profile.md)"
    )
    ax.set_xlabel("Dealer Absorption Surprise (actual - expected primary-dealer share, fraction)")
    ax.set_ylabel("Number of auctions")
    return _finish(
        fig,
        path,
        caption=(
            SOURCE_CAPTION
            + " Excludes the first calendar year of history (no strictly-prior training fold available yet)."
        ),
    )
