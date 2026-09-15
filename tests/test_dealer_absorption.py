import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.features.dealer_absorption import (
    REGIME_FEATURE_COL,
    DealerAbsorptionSurpriseModel,
    StressThreshold,
    add_regime_feature,
    compute_walk_forward_surprise,
    describe_percentile_thresholds,
)


def _synthetic_auctions(n_per_tenor=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2015-01-01")
    tenor_baseline = {"2-Year": 0.30, "10-Year": 0.20}
    for tenor, baseline in tenor_baseline.items():
        for i in range(n_per_tenor):
            rows.append(
                {
                    "auction_date": base_date + pd.Timedelta(days=30 * i)
                    + (pd.Timedelta(days=1) if tenor == "10-Year" else pd.Timedelta(days=0)),
                    "tenor": tenor,
                    "is_reopening": i % 3 == 0,
                    "offering_amt": 10e9 * (1 + 0.05 * rng.standard_normal()),
                    "primary_dealer_share": min(
                        max(baseline + 0.02 * rng.standard_normal(), 0.0), 1.0
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("auction_date").reset_index(drop=True)


def test_add_regime_feature_excludes_current_auction():
    df = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "tenor": ["2-Year"] * 4,
            "primary_dealer_share": [0.10, 0.20, 0.30, 0.40],
        }
    )
    out = add_regime_feature(df, window=8)
    assert pd.isna(out.loc[0, REGIME_FEATURE_COL])  # no prior history at all
    assert out.loc[1, REGIME_FEATURE_COL] == pytest.approx(0.10)
    assert out.loc[2, REGIME_FEATURE_COL] == pytest.approx((0.10 + 0.20) / 2)
    assert out.loc[3, REGIME_FEATURE_COL] == pytest.approx((0.10 + 0.20 + 0.30) / 3)


def test_regime_feature_is_unaffected_by_future_rows():
    full = _synthetic_auctions()
    early_cutoff = full["auction_date"].quantile(0.5, interpolation="lower")
    early_only = full.loc[full["auction_date"] <= early_cutoff].copy()

    out_full = add_regime_feature(full, window=8)
    out_early = add_regime_feature(early_only, window=8)

    merged = out_early[["auction_date", "tenor", REGIME_FEATURE_COL]].merge(
        out_full[["auction_date", "tenor", REGIME_FEATURE_COL]],
        on=["auction_date", "tenor"],
        suffixes=("_early", "_full"),
    )
    assert len(merged) == len(out_early)
    pd.testing.assert_series_equal(
        merged[f"{REGIME_FEATURE_COL}_early"],
        merged[f"{REGIME_FEATURE_COL}_full"],
        check_names=False,
    )


def test_model_recovers_group_mean_when_offering_and_regime_are_at_center():
    df = add_regime_feature(_synthetic_auctions(), window=8)
    model = DealerAbsorptionSurpriseModel().fit(df)

    row = pd.DataFrame(
        {
            "tenor": ["2-Year"],
            "is_reopening": [False],
            "offering_amt": [np.exp(model.offering_center_)],
            REGIME_FEATURE_COL: [model.regime_center_],
            "primary_dealer_share": [model.group_means_[("2-Year", False)]],
        }
    )
    out = model.transform(row)
    assert out.loc[0, "expected_dealer_share"] == pytest.approx(
        model.group_means_[("2-Year", False)], abs=1e-9
    )
    assert out.loc[0, "dealer_absorption_surprise"] == pytest.approx(0.0, abs=1e-9)


def test_transform_uses_frozen_training_parameters_for_unseen_group():
    df = add_regime_feature(_synthetic_auctions(), window=8)
    model = DealerAbsorptionSurpriseModel().fit(df)

    unseen_group_row = pd.DataFrame(
        {
            "tenor": ["30-Year"],  # never present in training data
            "is_reopening": [True],
            "offering_amt": [np.exp(model.offering_center_)],
            REGIME_FEATURE_COL: [model.regime_center_],
            "primary_dealer_share": [0.5],
        }
    )
    out = model.transform(unseen_group_row)
    # Falls back to the training-fold global mean, not a group mean
    # derived from this single unseen row.
    assert out.loc[0, "expected_dealer_share"] == pytest.approx(model.global_mean_, abs=1e-9)


def test_future_auctions_do_not_alter_historical_transformed_values():
    full = add_regime_feature(_synthetic_auctions(), window=8)
    cutoff = full["auction_date"].quantile(0.6, interpolation="lower")
    train = full.loc[full["auction_date"] <= cutoff]
    historical = full.loc[full["auction_date"] <= cutoff]
    future_appended = full  # includes rows strictly after cutoff too

    model = DealerAbsorptionSurpriseModel().fit(train)

    out_historical_only = model.transform(historical)
    out_with_future_rows = model.transform(future_appended)

    merged = out_historical_only[["auction_date", "tenor", "expected_dealer_share", "dealer_absorption_surprise"]].merge(
        out_with_future_rows[["auction_date", "tenor", "expected_dealer_share", "dealer_absorption_surprise"]],
        on=["auction_date", "tenor"],
        suffixes=("_alone", "_with_future"),
    )
    assert len(merged) == len(out_historical_only)
    pd.testing.assert_series_equal(
        merged["expected_dealer_share_alone"],
        merged["expected_dealer_share_with_future"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["dealer_absorption_surprise_alone"],
        merged["dealer_absorption_surprise_with_future"],
        check_names=False,
    )


def test_fit_raises_on_empty_training_data():
    empty = pd.DataFrame(columns=["tenor", "is_reopening", "offering_amt", "primary_dealer_share"])
    with pytest.raises(ValueError):
        DealerAbsorptionSurpriseModel().fit(empty)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        DealerAbsorptionSurpriseModel().transform(pd.DataFrame({"tenor": ["2-Year"]}))


def test_stress_threshold_ignores_test_period_values():
    train = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    threshold = StressThreshold(percentile=0.90).fit(train)
    threshold_before = threshold.threshold_

    # Passing a series containing extreme "test period" values to
    # transform must never change the already-fitted threshold.
    test_with_extremes = pd.concat([train, pd.Series([100.0, -100.0])], ignore_index=True)
    flags = threshold.transform(test_with_extremes)

    assert threshold.threshold_ == threshold_before
    assert flags.iloc[-2]  # 100.0 is flagged as stress...
    assert not flags.iloc[-1]  # ...but -100.0 is not, and the threshold itself didn't move


def test_stress_threshold_missing_values_stay_missing():
    threshold = StressThreshold(percentile=0.90).fit(pd.Series([0.1, 0.2, 0.3]))
    flags = threshold.transform(pd.Series([0.1, np.nan, 0.9]))
    assert not flags.iloc[0]
    assert pd.isna(flags.iloc[1])
    assert flags.iloc[2]


def test_stress_threshold_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        StressThreshold().transform(pd.Series([0.1]))


def test_describe_percentile_thresholds_is_monotonic_and_train_only():
    train = pd.Series(np.linspace(0, 1, 101))
    result = describe_percentile_thresholds(train, percentiles=(0.75, 0.90, 0.95))
    assert result["p75"]["threshold"] < result["p90"]["threshold"] < result["p95"]["threshold"]
    for entry in result.values():
        assert entry["n_train"] == 101


def _multi_year_auctions():
    df = add_regime_feature(_synthetic_auctions(n_per_tenor=40), window=8)
    return df


def test_walk_forward_surprise_excludes_first_year():
    df = _multi_year_auctions()
    first_year = df["auction_date"].dt.year.min()
    out = compute_walk_forward_surprise(df)
    assert (out["auction_date"].dt.year > first_year).all()
    assert out["dealer_absorption_surprise"].notna().all()


def test_walk_forward_surprise_matches_manual_expanding_fit():
    df = _multi_year_auctions()
    out = compute_walk_forward_surprise(df)
    years = sorted(df["auction_date"].dt.year.unique())
    target_year = years[-1]

    train = df.loc[df["auction_date"].dt.year < target_year]
    test = df.loc[df["auction_date"].dt.year == target_year]
    manual = DealerAbsorptionSurpriseModel().fit(train).transform(test)

    walk_forward_year = out.loc[out["auction_date"].dt.year == target_year].sort_values("auction_date")
    manual = manual.sort_values("auction_date")
    pd.testing.assert_series_equal(
        walk_forward_year["dealer_absorption_surprise"].reset_index(drop=True),
        manual["dealer_absorption_surprise"].reset_index(drop=True),
    )


def test_walk_forward_surprise_later_years_do_not_change_earlier_output():
    df = _multi_year_auctions()
    years = sorted(df["auction_date"].dt.year.unique())
    cutoff_year = years[len(years) // 2]

    truncated = df.loc[df["auction_date"].dt.year <= cutoff_year]
    out_truncated = compute_walk_forward_surprise(truncated)
    out_full = compute_walk_forward_surprise(df)

    merged = out_truncated[["auction_date", "tenor", "dealer_absorption_surprise"]].merge(
        out_full[["auction_date", "tenor", "dealer_absorption_surprise"]],
        on=["auction_date", "tenor"],
        suffixes=("_truncated", "_full"),
    )
    assert len(merged) == len(out_truncated)
    pd.testing.assert_series_equal(
        merged["dealer_absorption_surprise_truncated"],
        merged["dealer_absorption_surprise_full"],
        check_names=False,
    )
