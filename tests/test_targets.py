import pandas as pd
import pytest

from treasury_auction_stress.data.normalize import (
    nominal_coupon_subset,
    normalize_auctions,
    to_raw_dataframe,
)
from treasury_auction_stress.features.targets import (
    add_all_targets,
    add_bid_to_cover,
    add_bidder_shares,
    add_public_accepted_amount,
    reconcile_bid_to_cover,
)


def _nominal_df(sample_page_body):
    raw_df = to_raw_dataframe(sample_page_body["data"])
    normalized, _ = normalize_auctions(raw_df)
    return nominal_coupon_subset(normalized)


def test_public_accepted_amount_includes_fima_excludes_soma(sample_page_body):
    """Known example from the fixture: the 2024-01-23 2-year new issue.
    soma/fima add-ons are both zero for this row, so public_accepted_amount
    happens to equal total_accepted here -- see the synthetic tests below
    for rows where SOMA and FIMA differ and actually change the result.
    """
    df = add_public_accepted_amount(_nominal_df(sample_page_body))
    row = df.loc[df["cusip"] == "91282CJV4"].iloc[0]
    assert row["public_accepted_amount"] == 59460369400 + 539693200 + 0
    assert row["fed_reserve_addon_amount"] == 0


def test_dealer_share_uses_public_accepted_not_total_accepted(sample_page_body):
    df = add_bidder_shares(_nominal_df(sample_page_body))
    row = df.loc[df["cusip"] == "91282CJV4"].iloc[0]

    denominator = 59460369400 + 539693200 + 0  # comp + noncomp + fima (fima=0 here)
    expected_dealer_share = 8817250000 / denominator
    expected_direct_share = 11815074400 / denominator
    expected_indirect_share = 38828045000 / denominator
    expected_noncomp_share = 539693200 / denominator
    expected_fima_share = 0 / denominator

    assert row["primary_dealer_share"] == pytest.approx(expected_dealer_share)
    assert row["direct_bidder_share"] == pytest.approx(expected_direct_share)
    assert row["indirect_bidder_share"] == pytest.approx(expected_indirect_share)
    assert row["noncompetitive_share"] == pytest.approx(expected_noncomp_share)
    assert row["fima_share"] == pytest.approx(expected_fima_share)

    # The five categories partition public_accepted_amount exactly.
    total_share = (
        row["primary_dealer_share"]
        + row["direct_bidder_share"]
        + row["indirect_bidder_share"]
        + row["noncompetitive_share"]
        + row["fima_share"]
    )
    assert total_share == pytest.approx(1.0)


def test_soma_is_excluded_but_fima_is_included_in_share_denominator():
    """Synthetic auction with both a Fed (SOMA) rollover and a nonzero
    FIMA noncompetitive add-on, modeled on the real pattern verified in
    the Phase 2 acceptance review: Treasury's own published
    "Subtotal" (used for bid-to-cover) is Competitive + Noncompetitive +
    FIMA, excluding only SOMA. SOMA must not affect the share; FIMA must.
    """
    df = pd.DataFrame(
        {
            "primary_dealer_accepted": pd.array([20.0], dtype="Float64"),
            "direct_bidder_accepted": pd.array([10.0], dtype="Float64"),
            "indirect_bidder_accepted": pd.array([30.0], dtype="Float64"),
            "noncomp_accepted": pd.array([5.0], dtype="Float64"),
            "comp_accepted": pd.array([60.0], dtype="Float64"),
            "comp_tendered": pd.array([120.0], dtype="Float64"),
            "fima_noncomp_accepted": pd.array([15.0], dtype="Float64"),
            "total_accepted": pd.array([130.0], dtype="Float64"),  # 80 public + 50 Fed rollover
            "total_tendered": pd.array([190.0], dtype="Float64"),
            "bid_to_cover_ratio": pd.array([2.12], dtype="Float64"),
            "soma_accepted": pd.array([50.0], dtype="Float64"),
        }
    )
    out = add_bidder_shares(df)
    # public_accepted_amount = 60 (comp) + 5 (noncomp) + 15 (fima) = 80
    assert out.loc[0, "public_accepted_amount"] == 80.0
    assert out.loc[0, "primary_dealer_share"] == pytest.approx(20.0 / 80.0)
    assert out.loc[0, "fima_share"] == pytest.approx(15.0 / 80.0)
    # A version that wrongly excluded FIMA too would give 20/65, not 20/80.
    wrongly_excludes_fima = 20.0 / 65.0
    assert out.loc[0, "primary_dealer_share"] != pytest.approx(wrongly_excludes_fima)
    # A naive total_accepted-denominator share (including SOMA) would
    # have been 20/130, materially understating the true take-down share.
    naive_share_including_soma = 20.0 / 130.0
    assert out.loc[0, "primary_dealer_share"] > naive_share_including_soma


def test_missing_results_produce_missing_shares_not_zero(sample_page_body):
    """The pending 30-year reopening in the fixture (2026-09-10) has
    every result field as the API's "null" token -- shares must stay
    missing, never zero or silently inferred.
    """
    df = add_all_targets(_nominal_df(sample_page_body))
    pending_row = df.loc[~df["results_available"]].iloc[0]
    assert pd.isna(pending_row["public_accepted_amount"])
    assert pd.isna(pending_row["primary_dealer_share"])
    assert pd.isna(pending_row["direct_bidder_share"])
    assert pd.isna(pending_row["indirect_bidder_share"])
    assert pd.isna(pending_row["bid_to_cover_calculated"])


def test_zero_denominator_gives_missing_not_infinite():
    df = pd.DataFrame(
        {
            "primary_dealer_accepted": pd.array([0.0], dtype="Float64"),
            "direct_bidder_accepted": pd.array([0.0], dtype="Float64"),
            "indirect_bidder_accepted": pd.array([0.0], dtype="Float64"),
            "noncomp_accepted": pd.array([0.0], dtype="Float64"),
            "comp_accepted": pd.array([0.0], dtype="Float64"),
            "comp_tendered": pd.array([0.0], dtype="Float64"),
            "fima_noncomp_accepted": pd.array([0.0], dtype="Float64"),
            "total_accepted": pd.array([0.0], dtype="Float64"),
            "total_tendered": pd.array([0.0], dtype="Float64"),
            "bid_to_cover_ratio": pd.array([pd.NA], dtype="Float64"),
            "soma_accepted": pd.array([0.0], dtype="Float64"),
        }
    )
    out = add_all_targets(df)
    assert pd.isna(out.loc[0, "primary_dealer_share"])
    assert pd.isna(out.loc[0, "bid_to_cover_calculated"])


def test_bid_to_cover_reconciles_exactly(sample_page_body):
    """Known example: the 2023-01-11 10-year reopening. Treasury's own
    bid_to_cover_ratio (2.53) now reconciles EXACTLY (not just within a
    small tolerance) with the from-scratch calculation using
    (comp_tendered + noncomp_accepted + fima_noncomp_accepted) /
    public_accepted_amount -- see targets.py module docstring and
    the Phase 2 acceptance review for why FIMA must be included.
    """
    df = add_bid_to_cover(_nominal_df(sample_page_body))
    row = df.loc[df["cusip"] == "91282CFV8"].iloc[0]

    expected_calc = (80913151000 + 28502900 + 0) / (31971531000 + 28502900 + 0)
    assert row["bid_to_cover_calculated"] == pytest.approx(expected_calc)
    assert row["bid_to_cover_ratio"] == round(expected_calc, 2)


def test_bid_to_cover_reconciles_exactly_with_nonzero_fima():
    """Synthetic auction with a nonzero FIMA add-on, proving the
    reconciliation holds even when FIMA actually changes the ratio (the
    real fixture rows all happen to have FIMA=0, so this is needed to
    actually exercise the fix).
    """
    df = pd.DataFrame(
        {
            "primary_dealer_accepted": pd.array([20.0], dtype="Float64"),
            "direct_bidder_accepted": pd.array([10.0], dtype="Float64"),
            "indirect_bidder_accepted": pd.array([30.0], dtype="Float64"),
            "noncomp_accepted": pd.array([5.0], dtype="Float64"),
            "comp_accepted": pd.array([60.0], dtype="Float64"),
            "comp_tendered": pd.array([100.0], dtype="Float64"),
            "fima_noncomp_accepted": pd.array([15.0], dtype="Float64"),
            "total_accepted": pd.array([130.0], dtype="Float64"),
            "total_tendered": pd.array([170.0], dtype="Float64"),
            # Treasury's own methodology: (100 + 5 + 15) / (60 + 5 + 15) = 120/80 = 1.5
            "bid_to_cover_ratio": pd.array([1.5], dtype="Float64"),
            "soma_accepted": pd.array([50.0], dtype="Float64"),
        }
    )
    out = add_bid_to_cover(df)
    assert out.loc[0, "bid_to_cover_calculated"] == pytest.approx(1.5)
    assert out.loc[0, "bid_to_cover_ratio"] == round(out.loc[0, "bid_to_cover_calculated"], 2)


def test_reconcile_bid_to_cover_summary_shape(sample_page_body):
    summary = reconcile_bid_to_cover(_nominal_df(sample_page_body))
    assert summary["n_compared"] >= 1
    assert summary["max_abs_diff"] >= 0
    assert summary["n_within_0_01"] + summary["n_over_0_02"] <= summary["n_compared"]


def test_primary_dealer_share_never_negative_or_above_one(sample_page_body):
    df = add_all_targets(_nominal_df(sample_page_body))
    settled = df.loc[df["results_available"], "primary_dealer_share"].dropna()
    assert (settled >= 0).all()
    assert (settled <= 1.0 + 1e-9).all()
