from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treasury_auction_stress.data.rtdsm_schema import VARIABLE_REGISTRY
from treasury_auction_stress.features.auction_cutoffs import (
    ANNOUNCEMENT_CUTOFF_COL,
    PRE_AUCTION_CUTOFF_COL,
    add_cutoff_dates,
)
from treasury_auction_stress.features.feature_manifest import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
    parse_entries,
)
from treasury_auction_stress.features.feature_matrix import (
    AUCTION_KEY_COL,
    SourceTables,
    build_audit_table,
    build_join_tables,
    build_pre_auction_update_table,
    build_predictor_matrix,
    build_target_table,
    make_auction_key,
    validate_audit_table,
    validate_pre_auction_update_table,
    validate_predictor_matrix,
    validate_target_table,
)
from treasury_auction_stress.features.leakage_audit import (
    assert_cross_cutoff_monotonicity,
    assert_safe_availability_never_exceeds_cutoff,
    forbidden_field_scan,
    future_row_invariance_test,
    input_order_invariance_test,
    outcome_poisoning_test,
)
from treasury_auction_stress.features.phase4d_source_joins import (
    FORBIDDEN_RESULT_COLUMNS,
)

TENORS = ("2-Year", "5-Year", "10-Year", "3-Year")


def _synthetic_sample(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        tenor = TENORS[i % len(TENORS)]
        ann = pd.Timestamp("2020-01-01") + pd.Timedelta(days=14 * i)
        auc = ann + pd.Timedelta(days=6)
        rows.append(
            {
                "cusip": f"CUSIP{i:03d}",
                "tenor": tenor,
                "announcemt_date": ann,
                "auction_date": auc,
                "is_reopening": bool(i % 3 == 0),
                "offering_amt": 30_000_000_000.0 + i * 1e9,
                # A full, plausible set of forbidden result fields, for
                # the outcome-poisoning test.
                "high_yield": 4.0 + i * 0.01,
                "total_accepted": 30_000_000_000.0 + i,
                "total_tendered": 90_000_000_000.0 + i,
                "bid_to_cover_ratio": 3.0 + i * 0.01,
                "comp_accepted": 25_000_000_000.0,
                "comp_tendered": 85_000_000_000.0,
                "noncomp_accepted": 100_000_000.0,
                "fima_noncomp_accepted": 200_000_000.0,
                "soma_accepted": 500_000_000.0,
                "primary_dealer_accepted": 10_000_000_000.0,
                "direct_bidder_accepted": 8_000_000_000.0,
                "indirect_bidder_accepted": 7_000_000_000.0,
                "results_available": True,
            }
        )
    df = pd.DataFrame(rows)
    return add_cutoff_dates(df)


def _synthetic_dealer_wide(n_weeks: int = 60, start: str = "2019-01-02") -> pd.DataFrame:
    obs = pd.date_range(start, periods=n_weeks, freq="7D")
    pub = obs + pd.Timedelta(days=1)
    safe = pub + pd.Timedelta(days=1)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "observation_date": obs,
            "publication_date": pub,
            "publication_safe_available_date": safe,
            "dealer_net_position_bills": rng.normal(1000, 50, n_weeks),
            "dealer_net_position_coupons_le_2y": rng.normal(500, 20, n_weeks),
            "dealer_net_position_coupons_2y_3y": rng.normal(400, 20, n_weeks),
            "dealer_net_position_coupons_3y_6y": rng.normal(600, 20, n_weeks),
            "dealer_net_position_coupons_6y_7y": rng.normal(300, 20, n_weeks),
            "dealer_net_position_coupons_7y_11y": rng.normal(700, 20, n_weeks),
            "dealer_net_position_coupons_11y_21y": rng.normal(200, 20, n_weeks),
            "dealer_net_position_coupons_gt_21y": rng.normal(150, 20, n_weeks),
            "dealer_net_position_total_ex_tips": rng.normal(5000, 100, n_weeks),
            "dealer_repo_treasury_ex_tips_total": rng.normal(2000, 50, n_weeks),
            "dealer_reverse_repo_treasury_ex_tips_total": rng.normal(1800, 50, n_weeks),
            "dealer_securities_borrowed_treasury_ex_tips_total": rng.normal(300, 20, n_weeks),
            "dealer_securities_lent_treasury_ex_tips_total": rng.normal(250, 20, n_weeks),
            "dealer_transaction_volume_total_ex_tips": rng.normal(9000, 200, n_weeks),
            "dealer_fails_to_deliver_treasury_ex_tips": rng.normal(100, 10, n_weeks),
            "dealer_fails_to_receive_treasury_ex_tips": rng.normal(90, 10, n_weeks),
        }
    )


def _synthetic_rates_wide(n_days: int = 400, start: str = "2019-01-02") -> pd.DataFrame:
    rate_date = pd.bdate_range(start, periods=n_days)
    pub = rate_date
    safe = rate_date + pd.Timedelta(days=1)
    rng = np.random.default_rng(1)
    base = {
        "rate_date": rate_date,
        "publication_date": pub,
        "publication_safe_available_date": safe,
    }
    for label, level in (("2 Yr", 1.5), ("3 Yr", 1.6), ("5 Yr", 1.8), ("7 Yr", 2.0), ("10 Yr", 2.2), ("20 Yr", 2.5), ("30 Yr", 2.7)):
        base[label] = level + rng.normal(0, 0.02, n_days).cumsum() * 0.01
    return pd.DataFrame(base)


def _synthetic_cftc_wide(n_weeks: int = 60, start: str = "2019-01-01") -> pd.DataFrame:
    report_date = pd.date_range(start, periods=n_weeks, freq="7D")
    safe = report_date + pd.Timedelta(days=4)
    rng = np.random.default_rng(2)
    codes = ("042601", "044601", "043602", "020601")
    data = {"report_date": report_date, "publication_safe_available_date": safe}
    for code in codes:
        data[f"{code}__dealer_net_contracts"] = rng.normal(1000, 50, n_weeks)
        data[f"{code}__asset_mgr_net_contracts"] = rng.normal(500, 50, n_weeks)
        data[f"{code}__lev_money_net_contracts"] = rng.normal(-200, 50, n_weeks)
        data[f"{code}__other_rept_net_contracts"] = rng.normal(100, 20, n_weeks)
        data[f"{code}__nonrept_net_contracts"] = rng.normal(50, 10, n_weeks)
        data[f"{code}__open_interest_all"] = rng.normal(500000, 1000, n_weeks)
        data[f"{code}__dealer_long_pct_oi"] = rng.uniform(0.1, 0.3, n_weeks)
        data[f"{code}__dealer_short_pct_oi"] = rng.uniform(0.1, 0.3, n_weeks)
        data[f"{code}__dealer_spread_pct_oi"] = rng.uniform(0.0, 0.1, n_weeks)
        data[f"{code}__conc_gross_le_4_tdr_long"] = rng.uniform(0, 1, n_weeks)
        data[f"{code}__conc_gross_le_4_tdr_short"] = rng.uniform(0, 1, n_weeks)
        data[f"{code}__conc_gross_le_8_tdr_long"] = rng.uniform(0, 1, n_weeks)
        data[f"{code}__conc_gross_le_8_tdr_short"] = rng.uniform(0, 1, n_weeks)
        data[f"{code}__dealer_net_contracts_chg_1w"] = rng.normal(0, 10, n_weeks)
        data[f"{code}__dealer_net_contracts_chg_4w"] = rng.normal(0, 20, n_weeks)
    return pd.DataFrame(data)


def _synthetic_rtdsm() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    vintage_indices, snapshots = {}, {}
    for variable in VARIABLE_REGISTRY:
        m = variable.mnemonic
        vintage_indices[m] = pd.DataFrame(
            {
                "vintage_label": ["19Q4", "20Q1", "20Q2"],
                "vintage_year": [2019, 2020, 2020],
                "vintage_period": [4, 1, 2],
                "nominal_publication_date": pd.to_datetime(["2019-11-15", "2020-02-15", "2020-05-15"]),
                "publication_safe_available_date": pd.to_datetime(["2019-11-18", "2020-02-18", "2020-05-18"]),
                "availability_precision": ["test", "test", "test"],
            }
        )
        snapshots[m] = pd.DataFrame(
            {
                "vintage_label": ["19Q4", "20Q1", "20Q2"],
                "current_observation_date": pd.to_datetime(["2019-10-01", "2020-01-01", "2020-04-01"]),
                "current_value": [100.0, 101.0, 99.0],
                "yoy_observation_date": pd.to_datetime(["2018-10-01", "2019-01-01", "2019-04-01"]),
                "yoy_value": [98.0, 99.0, 97.0],
            }
        )
    return vintage_indices, snapshots


@pytest.fixture(scope="module")
def entries():
    return parse_entries(load_contract(DEFAULT_CONTRACT_PATH))


def _dealer_feat(**kwargs) -> pd.DataFrame:
    """`_synthetic_dealer_wide` is the raw wide table; the real
    pipeline always runs it through `build_dealer_feature_table` first
    (harmonized buckets, trailing changes, rolling z-scores) before it
    reaches `SourceTables` -- mirrored here so fixtures match production.
    """
    from treasury_auction_stress.features.dealer_candidate_features import (
        build_dealer_feature_table,
    )

    return build_dealer_feature_table(_synthetic_dealer_wide(**kwargs))


@pytest.fixture()
def synthetic_tables() -> SourceTables:
    return SourceTables(
        sample=_synthetic_sample(),
        dealer_wide=_dealer_feat(),
        rates_wide=_synthetic_rates_wide(),
        cftc_wide=_synthetic_cftc_wide(),
        rtdsm_vintage_indices=_synthetic_rtdsm()[0],
        rtdsm_snapshots=_synthetic_rtdsm()[1],
    )


@pytest.fixture()
def joined(synthetic_tables):
    return build_join_tables(synthetic_tables)


# ---------------------------------------------------------------------------
# Row and key integrity
# ---------------------------------------------------------------------------


def test_predictor_matrix_one_row_per_auction_no_duplicates(joined, entries):
    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    assert len(matrix) == len(joined["sample"])
    assert matrix[AUCTION_KEY_COL].is_unique
    validate_predictor_matrix(matrix, entries, expected_rows=len(joined["sample"]))


def test_predictor_matrix_deterministic_column_order(joined, entries):
    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    predictor_cols = [c for c in matrix.columns if c not in ("auction_key", "cusip", "auction_date", "announcemt_date", ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL)]
    assert predictor_cols == sorted(predictor_cols)


def test_both_matrices_share_identical_column_schema(joined, entries):
    ann = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    pre = build_predictor_matrix(cutoff_name="pre_auction", joined=joined, entries=entries)
    assert list(ann.columns) == list(pre.columns)
    assert (ann.dtypes.astype(str) == pre.dtypes.astype(str)).all()


def test_no_auction_dropped(joined, entries):
    ann = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    assert set(ann[AUCTION_KEY_COL]) == set(make_auction_key(joined["sample"]))


# ---------------------------------------------------------------------------
# Forbidden fields
# ---------------------------------------------------------------------------


def test_predictor_matrix_contains_no_forbidden_result_fields(joined, entries):
    for cutoff_name in ("announcement", "pre_auction"):
        matrix = build_predictor_matrix(cutoff_name=cutoff_name, joined=joined, entries=entries)
        scan = forbidden_field_scan(list(matrix.columns))
        assert scan["exact_matches"] == []
        assert scan["pattern_matches"] == []


def test_forbidden_field_scan_catches_a_renamed_result_field():
    scan = forbidden_field_scan(["tenor", "primary_dealr_accepted_renamed_typo_but_still_yield_word", "matched_tenor_par_yield_percent"])
    # The allowlisted legitimate yield feature must not be flagged...
    assert "matched_tenor_par_yield_percent" not in scan["pattern_matches"]
    # ...but a field whose name still contains a forbidden substring must be.
    assert any("yield" in c.lower() for c in scan["pattern_matches"])


# ---------------------------------------------------------------------------
# Availability boundaries
# ---------------------------------------------------------------------------


def test_safe_availability_never_exceeds_cutoff(joined):
    results = assert_safe_availability_never_exceeds_cutoff(joined)
    assert results, "expected at least one source/cutoff pair to be checked"
    for key, r in results.items():
        assert r["n_violations"] == 0, f"{key}: {r}"


def test_availability_boundary_equality_at_safe_date_selects_the_release():
    """A cutoff exactly equal to a release's own safe-available date
    must select it (equality allowed, per docs/point_in_time_rules.md).
    """
    from treasury_auction_stress.features.dealer_join import as_of_join

    dealer_wide = _synthetic_dealer_wide(n_weeks=4, start="2020-01-01")
    row = dealer_wide.iloc[[1]]
    cutoff = row["publication_safe_available_date"].iloc[0]
    auctions = pd.DataFrame({"cusip": ["X"], "tenor": ["2-Year"], "auction_date": [cutoff + pd.Timedelta(days=1)]})
    auctions["announcement_cutoff_date"] = cutoff
    joined_result = as_of_join(auctions, dealer_wide, cutoff_col="announcement_cutoff_date")
    assert joined_result["observation_date"].iloc[0] == row["observation_date"].iloc[0]


def test_availability_boundary_one_day_before_does_not_select_release():
    from treasury_auction_stress.features.dealer_join import as_of_join

    dealer_wide = _synthetic_dealer_wide(n_weeks=4, start="2020-01-01")
    row = dealer_wide.iloc[[1]]
    cutoff = row["publication_safe_available_date"].iloc[0] - pd.Timedelta(days=1)
    auctions = pd.DataFrame({"cusip": ["X"], "tenor": ["2-Year"], "auction_date": [cutoff + pd.Timedelta(days=1)]})
    auctions["announcement_cutoff_date"] = cutoff
    joined_result = as_of_join(auctions, dealer_wide, cutoff_col="announcement_cutoff_date")
    assert joined_result["observation_date"].iloc[0] != row["observation_date"].iloc[0]


# ---------------------------------------------------------------------------
# Cross-cutoff monotonicity
# ---------------------------------------------------------------------------


def test_cross_cutoff_monotonicity_holds(joined):
    results = assert_cross_cutoff_monotonicity(joined)
    for key, r in results.items():
        assert r["n_violations"] == 0, f"{key}: {r}"


# ---------------------------------------------------------------------------
# Future-row invariance / no future backfill
# ---------------------------------------------------------------------------


def test_future_row_invariance(synthetic_tables, entries):
    def build(sample):
        tables = SourceTables(
            sample=sample,
            dealer_wide=synthetic_tables.dealer_wide,
            rates_wide=synthetic_tables.rates_wide,
            cftc_wide=synthetic_tables.cftc_wide,
            rtdsm_vintage_indices=synthetic_tables.rtdsm_vintage_indices,
            rtdsm_snapshots=synthetic_tables.rtdsm_snapshots,
        )
        j = build_join_tables(tables)
        return build_predictor_matrix(cutoff_name="announcement", joined=j, entries=entries)

    future_row = synthetic_tables.sample.iloc[[0]].copy()
    future_row["cusip"] = "FUTURE99"
    future_row["auction_date"] = pd.Timestamp("2099-01-05")
    future_row["announcemt_date"] = pd.Timestamp("2098-12-29")
    future_row = add_cutoff_dates(future_row.drop(columns=[ANNOUNCEMENT_CUTOFF_COL, PRE_AUCTION_CUTOFF_COL]))

    result = future_row_invariance_test(build, synthetic_tables.sample, future_row)
    assert result["identical_for_historical_rows"], result.get("error")


# ---------------------------------------------------------------------------
# Outcome-poisoning invariance
# ---------------------------------------------------------------------------


def test_outcome_poisoning_invariance(synthetic_tables, entries):
    def build(sample):
        tables = SourceTables(
            sample=sample,
            dealer_wide=synthetic_tables.dealer_wide,
            rates_wide=synthetic_tables.rates_wide,
            cftc_wide=synthetic_tables.cftc_wide,
            rtdsm_vintage_indices=synthetic_tables.rtdsm_vintage_indices,
            rtdsm_snapshots=synthetic_tables.rtdsm_snapshots,
        )
        j = build_join_tables(tables)
        return build_predictor_matrix(cutoff_name="announcement", joined=j, entries=entries)

    present = tuple(c for c in FORBIDDEN_RESULT_COLUMNS if c in synthetic_tables.sample.columns)
    assert present, "fixture must include at least one forbidden result column to poison"
    result = outcome_poisoning_test(build, synthetic_tables.sample, outcome_columns=present)
    assert result["identical"], result.get("error")


# ---------------------------------------------------------------------------
# Input-order invariance
# ---------------------------------------------------------------------------


def test_input_order_invariance(synthetic_tables, entries):
    def build(sample):
        tables = SourceTables(
            sample=sample,
            dealer_wide=synthetic_tables.dealer_wide,
            rates_wide=synthetic_tables.rates_wide,
            cftc_wide=synthetic_tables.cftc_wide,
            rtdsm_vintage_indices=synthetic_tables.rtdsm_vintage_indices,
            rtdsm_snapshots=synthetic_tables.rtdsm_snapshots,
        )
        j = build_join_tables(tables)
        return build_predictor_matrix(cutoff_name="announcement", joined=j, entries=entries)

    result = input_order_invariance_test(build, synthetic_tables.sample)
    assert result["identical"], result.get("error")


def test_per_source_input_order_invariance_covers_all_five_sources_both_cutoffs(synthetic_tables, entries):
    from treasury_auction_stress.features.leakage_audit import (
        PERTURBABLE_SOURCES,
        per_source_input_order_invariance,
    )

    results = per_source_input_order_invariance(synthetic_tables, entries)
    assert set(results) == set(PERTURBABLE_SOURCES)
    for source_name, per_cutoff in results.items():
        assert set(per_cutoff) == {"announcement", "pre_auction"}
        for cutoff_name, r in per_cutoff.items():
            assert r["identical"], f"{source_name}/{cutoff_name}: {r.get('error')}"
            assert r["n_rows_perturbed"] > 0


def test_per_source_future_row_invariance_covers_all_five_sources_both_cutoffs(synthetic_tables, entries):
    from treasury_auction_stress.features.leakage_audit import (
        PERTURBABLE_SOURCES,
        per_source_future_row_invariance,
    )

    results = per_source_future_row_invariance(synthetic_tables, entries)
    assert set(results) == set(PERTURBABLE_SOURCES)
    for source_name, per_cutoff in results.items():
        assert set(per_cutoff) == {"announcement", "pre_auction"}
        for cutoff_name, r in per_cutoff.items():
            assert r["identical_for_historical_rows"], f"{source_name}/{cutoff_name}: {r.get('error')}"


# ---------------------------------------------------------------------------
# Matrix / target separation
# ---------------------------------------------------------------------------


def test_predictor_matrix_builds_without_target_table(joined, entries):
    import inspect

    sig = inspect.signature(build_predictor_matrix)
    assert "targets" not in sig.parameters
    assert "target_table" not in sig.parameters


def test_targets_merge_only_as_an_explicit_later_step(joined, entries):
    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    targets = build_target_table(joined["sample"])
    validate_target_table(targets, entries, expected_rows=len(joined["sample"]))

    assert set(matrix.columns).isdisjoint(set(targets.columns) - {AUCTION_KEY_COL})
    merged = matrix.merge(targets, on=AUCTION_KEY_COL, how="left")
    assert len(merged) == len(matrix)
    assert "primary_dealer_share" in merged.columns


# ---------------------------------------------------------------------------
# Key-based alignment (acceptance-review addition, Issue 1)
# ---------------------------------------------------------------------------


def test_every_join_table_carries_a_validated_auction_key(joined):
    from treasury_auction_stress.features.feature_matrix import (
        AUCTION_KEY_COL,
        make_auction_key,
    )

    canonical = set(joined["sample"][AUCTION_KEY_COL])
    for table_key in ("auction_structure", "dealer", "rates", "cftc", "rtdsm"):
        table = joined[table_key]
        assert AUCTION_KEY_COL in table.columns, f"{table_key} missing auction_key"
        assert not table[AUCTION_KEY_COL].duplicated().any(), f"{table_key} has duplicate auction keys"
        assert set(table[AUCTION_KEY_COL]) == canonical, f"{table_key} key set disagrees with the canonical sample"
        # The stored key must actually match what cusip/auction_date compute to.
        pd.testing.assert_series_equal(
            table[AUCTION_KEY_COL].reset_index(drop=True),
            make_auction_key(table).reset_index(drop=True),
            check_names=False,
        )


def test_shuffling_each_intermediate_join_table_does_not_misalign_the_matrix(joined, entries):
    """Independently shuffle each of the five intermediate join tables
    (never the auction sample itself) and prove both predictor matrices
    and both audit tables are bit-identical to the unshuffled baseline
    -- direct proof that combination is by auction_key, not row
    position.
    """
    rng = np.random.default_rng(42)
    baselines = {
        ("announcement", "matrix"): build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries),
        ("pre_auction", "matrix"): build_predictor_matrix(cutoff_name="pre_auction", joined=joined, entries=entries),
        ("announcement", "audit"): build_audit_table(cutoff_name="announcement", joined=joined, entries=entries),
        ("pre_auction", "audit"): build_audit_table(cutoff_name="pre_auction", joined=joined, entries=entries),
    }

    for table_key in ("auction_structure", "dealer", "rates", "cftc", "rtdsm"):
        table = joined[table_key]
        shuffled_joined = dict(joined)
        shuffled_joined[table_key] = table.iloc[rng.permutation(len(table))].reset_index(drop=True)

        for cutoff_name in ("announcement", "pre_auction"):
            shuffled_matrix = build_predictor_matrix(cutoff_name=cutoff_name, joined=shuffled_joined, entries=entries)
            pd.testing.assert_frame_equal(
                baselines[(cutoff_name, "matrix")], shuffled_matrix, obj=f"matrix after shuffling {table_key} ({cutoff_name})"
            )
            shuffled_audit = build_audit_table(cutoff_name=cutoff_name, joined=shuffled_joined, entries=entries)
            pd.testing.assert_frame_equal(
                baselines[(cutoff_name, "audit")], shuffled_audit, obj=f"audit after shuffling {table_key} ({cutoff_name})"
            )


def test_swapped_keys_are_honored_not_silently_repositioned(joined, entries):
    """Deliberately swap which two rows' cusip/auction_date (hence
    auction_key) values are attached to a source table's own feature
    values, while keeping the row count and the overall key SET
    unchanged (a corrupted-but-internally-consistent source table).
    Proves the merge is genuinely key-driven: if combination were still
    positional, this swap (which does not change row order) would have
    NO effect on the output; because it is key-driven, the swapped
    values must follow their (swapped) key to the correct output row.
    """
    from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL

    predictor_col = "dealer_repo_treasury_ex_tips_total"  # tenor-independent, always populated in the fixture
    dealer = joined["dealer"]
    i, j = 0, 1
    key_i, key_j = dealer.loc[i, AUCTION_KEY_COL], dealer.loc[j, AUCTION_KEY_COL]
    ann_col = f"announcement_cutoff_date__{predictor_col}"
    value_i, value_j = dealer.loc[i, ann_col], dealer.loc[j, ann_col]
    assert value_i != value_j, "fixture rows must have distinguishable values for this test to be meaningful"

    swapped = dealer.copy()
    id_cols = ["cusip", "auction_date", AUCTION_KEY_COL]
    swapped.loc[i, id_cols] = dealer.loc[j, id_cols].to_numpy()
    swapped.loc[j, id_cols] = dealer.loc[i, id_cols].to_numpy()

    swapped_joined = dict(joined)
    swapped_joined["dealer"] = swapped

    result = build_predictor_matrix(cutoff_name="announcement", joined=swapped_joined, entries=entries)
    result_indexed = result.set_index(AUCTION_KEY_COL)[predictor_col]

    # key_j's output row must now carry what was originally key_i's value (and vice versa).
    assert result_indexed.loc[key_j] == value_i
    assert result_indexed.loc[key_i] == value_j


def test_duplicate_key_in_a_source_table_raises():
    """A source table with the right row count but a duplicated
    auction key (and, necessarily, a missing one) must raise loudly,
    never silently drop or misalign a row.
    """
    from treasury_auction_stress.features.feature_matrix import (
        AUCTION_KEY_COL,
        _merge_selected,
        _with_auction_key,
    )

    base = pd.DataFrame({AUCTION_KEY_COL: ["A_2020-01-01", "B_2020-01-02"], "x": [1, 2]})
    corrupted = pd.DataFrame(
        {"cusip": ["A", "A"], "auction_date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "value": [10.0, 20.0]}
    )
    corrupted = _with_auction_key(corrupted)
    with pytest.raises(ValueError, match="duplicate auction key"):
        _merge_selected(base, corrupted, source_columns=["value"], rename_to=None, table_label="corrupted")


def test_missing_key_in_a_source_table_raises():
    from treasury_auction_stress.features.feature_matrix import (
        AUCTION_KEY_COL,
        _merge_selected,
        _with_auction_key,
    )

    base = pd.DataFrame({AUCTION_KEY_COL: ["A_2020-01-01", "B_2020-01-02"], "x": [1, 2]})
    incomplete = pd.DataFrame({"cusip": ["A"], "auction_date": pd.to_datetime(["2020-01-01"]), "value": [10.0]})
    incomplete = _with_auction_key(incomplete)
    with pytest.raises(ValueError, match="key set does not match"):
        _merge_selected(base, incomplete, source_columns=["value"], rename_to=None, table_label="incomplete")


# ---------------------------------------------------------------------------
# Audit tables
# ---------------------------------------------------------------------------


def test_audit_table_keyed_one_to_one_with_matrix(joined, entries):
    for cutoff_name in ("announcement", "pre_auction"):
        matrix = build_predictor_matrix(cutoff_name=cutoff_name, joined=joined, entries=entries)
        audit = build_audit_table(cutoff_name=cutoff_name, joined=joined, entries=entries)
        validate_audit_table(audit, matrix, cutoff_name=cutoff_name)
        assert not set(audit.columns) & set(matrix.columns) - {AUCTION_KEY_COL, "cusip", "auction_date"}


def test_audit_table_contains_no_forbidden_result_fields(joined, entries):
    for cutoff_name in ("announcement", "pre_auction"):
        audit = build_audit_table(cutoff_name=cutoff_name, joined=joined, entries=entries)
        scan = forbidden_field_scan(list(audit.columns))
        assert scan["exact_matches"] == []


# ---------------------------------------------------------------------------
# Pre-auction update table
# ---------------------------------------------------------------------------


def test_pre_auction_update_table_keyed_to_matrix(joined, entries):
    ann = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    pre = build_predictor_matrix(cutoff_name="pre_auction", joined=joined, entries=entries)
    updates = build_pre_auction_update_table(joined=joined, announcement_matrix=ann, pre_auction_matrix=pre)
    validate_pre_auction_update_table(updates, ann)
    assert (updates["cutoff_gap_calendar_days"] >= 0).all()


def test_pre_auction_update_table_distinguishes_no_update_from_unavailable():
    """A source with zero coverage at either cutoff must report False
    (no update occurred), never True and never a fabricated zero-valued
    observation.
    """
    sample = _synthetic_sample(n=2)
    # Push both cutoffs before any dealer_wide coverage exists.
    tables = SourceTables(
        sample=sample,
        dealer_wide=_dealer_feat(start="2021-06-01"),  # starts after both cutoffs
        rates_wide=_synthetic_rates_wide(),
        cftc_wide=_synthetic_cftc_wide(),
        rtdsm_vintage_indices=_synthetic_rtdsm()[0],
        rtdsm_snapshots=_synthetic_rtdsm()[1],
    )
    j = build_join_tables(tables)
    ann_matched = j["dealer"][f"{ANNOUNCEMENT_CUTOFF_COL}__dealer_join_matched"]
    assert not ann_matched.any()

    contract = load_contract(DEFAULT_CONTRACT_PATH)
    entries_local = parse_entries(contract)
    ann = build_predictor_matrix(cutoff_name="announcement", joined=j, entries=entries_local)
    pre = build_predictor_matrix(cutoff_name="pre_auction", joined=j, entries=entries_local)
    updates = build_pre_auction_update_table(joined=j, announcement_matrix=ann, pre_auction_matrix=pre)
    assert not updates["dealer_stats_new_release_between_cutoffs"].any()


def test_real_data_shows_all_four_distinguishable_update_states(entries):
    """Acceptance-review addition (Issue 6): 'no new release', 'new
    release with unchanged value', 'structurally unavailable', and
    'missing value' must remain distinguishable. Verified directly
    against this project's real, full pipeline output -- not asserted
    in the abstract.
    """
    processed = Path("data/processed")
    required_updates = processed / "pre_auction_information_updates.parquet"
    required_audit = processed / "feature_audit_announcement.parquet"
    if not (required_updates.exists() and required_audit.exists()):
        pytest.skip("processed Phase 5 outputs not present in this environment")

    updates = pd.read_parquet(required_updates)
    audit_ann = pd.read_parquet(required_audit)

    # State 1: "no new release" -- the source's own release identity is
    # unchanged between the two cutoffs.
    no_update = ~updates["treasury_rates_new_release_between_cutoffs"]
    assert no_update.sum() > 0, "expected at least one real auction with no rate release between cutoffs"

    # State 2: "new release, value unchanged" -- the release identity
    # DID change, but the tracked feature's own value is exactly the
    # same (never conflated with 'no update occurred').
    updated_but_same_value = updates["treasury_rates_new_release_between_cutoffs"] & (
        updates["slope_2s10s_bps_change_between_cutoffs"] == 0
    )
    assert updated_but_same_value.sum() > 0, "expected at least one real auction with a new rate release but an unchanged slope"
    assert no_update.sum() != updated_but_same_value.sum() or not (no_update & updated_but_same_value).any()

    # State 3: "structurally unavailable" -- 3/7/20-Year auctions have
    # no direct CFTC contract at all, so the tracked CFTC change
    # feature is NaN, never a fabricated zero.
    structurally_unavailable = updates["matched_dealer_net_contracts_change_between_cutoffs"].isna()
    assert structurally_unavailable.sum() > 0

    # State 4: "missing value" (a genuine source-reported gap, distinct
    # from structural unavailability) -- visible in the audit table's
    # own missing_reason taxonomy, never collapsed into state 3's
    # reason or silently zero-filled.
    missing_reason_col = "dealer_stats__dealer_securities_lent_treasury_ex_tips_total_missing_reason"
    assert missing_reason_col in audit_ann.columns
    reasons = audit_ann[missing_reason_col].dropna().str.extract(r"^([a-z_]+)")[0]
    assert "source_reported_missing_value_for_nearest_available_release" in set(reasons)
    assert "series_regime_not_yet_started" in set(reasons)
    # The two reasons must be genuinely distinct categories, not merged.
    assert (reasons == "source_reported_missing_value_for_nearest_available_release").sum() > 0
    assert (reasons == "series_regime_not_yet_started").sum() > 0


# ---------------------------------------------------------------------------
# Point-in-time boundary recheck against real data (acceptance-review Issue 7)
# ---------------------------------------------------------------------------


def test_real_data_holiday_shifted_friday_publication_never_lands_on_a_weekend_safe_date():
    """Direct, real-data proof of the Phase 3 safe-date fix: every
    historical week whose dealer-statistics publication_date was
    shifted onto a Friday by a Thursday federal holiday (verified: 41
    such weeks exist in this project's real data, e.g. the week of
    2001-11-21, Thanksgiving) has a publication_safe_available_date
    that is NEVER a Saturday or Sunday, across the ENTIRE dataset.
    """
    processed = Path("data/processed")
    if not (processed / "dealer_stats_long.parquet").exists():
        pytest.skip("processed dealer stats not present in this environment")

    long_df = pd.read_parquet(processed / "dealer_stats_long.parquet")
    long_df = long_df.drop_duplicates(subset=["observation_date"])[
        ["observation_date", "publication_date", "publication_safe_available_date"]
    ]
    friday_pubs = long_df.loc[long_df["publication_date"].dt.dayofweek == 4]
    assert len(friday_pubs) > 0, "expected at least one real holiday-shifted Friday publication_date"
    # A concrete, named example: the 2001-11-21 (Thanksgiving) week.
    thanksgiving_2001 = friday_pubs.loc[friday_pubs["observation_date"] == pd.Timestamp("2001-11-21")]
    assert len(thanksgiving_2001) == 1
    assert thanksgiving_2001["publication_date"].iloc[0] == pd.Timestamp("2001-11-23")  # Friday
    assert thanksgiving_2001["publication_safe_available_date"].iloc[0] == pd.Timestamp("2001-11-26")  # Monday

    weekend_safe_dates = long_df.loc[long_df["publication_safe_available_date"].dt.dayofweek >= 5]
    assert len(weekend_safe_dates) == 0


def test_real_data_no_direct_cftc_contract_is_always_nan_never_zero_for_3_7_20_year():
    processed = Path("data/processed")
    if not (processed / "feature_matrix_announcement.parquet").exists():
        pytest.skip("processed Phase 5 outputs not present in this environment")

    ann = pd.read_parquet(processed / "feature_matrix_announcement.parquet")
    for tenor in ("3-Year", "7-Year", "20-Year"):
        sub = ann.loc[ann["tenor"] == tenor, "matched_dealer_net_contracts"]
        assert len(sub) > 0
        assert sub.isna().all(), f"{tenor} should have no direct CFTC contract at all (100% NaN)"
    for tenor in ("2-Year", "10-Year"):
        sub = ann.loc[ann["tenor"] == tenor, "matched_dealer_net_contracts"]
        assert sub.notna().all(), f"{tenor} has a direct CFTC contract and should have full coverage"


def test_real_data_rtdsm_source_gap_carries_forward_the_real_stale_value_never_zero():
    """The documented October 2025 RUC government-shutdown-adjacent gap
    (Phase 4) must still be honestly flagged in the Phase 5 audit
    table, and the corresponding predictor-matrix level must be the
    real, stale, carried-forward value -- never a fabricated zero.
    """
    processed = Path("data/processed")
    required = [processed / "feature_matrix_announcement.parquet", processed / "feature_audit_announcement.parquet"]
    if not all(p.exists() for p in required):
        pytest.skip("processed Phase 5 outputs not present in this environment")

    ann = pd.read_parquet(processed / "feature_matrix_announcement.parquet")
    audit_ann = pd.read_parquet(processed / "feature_audit_announcement.parquet")
    gap_col = "rtdsm_macro__RUC_source_gap_detected"
    merged = ann[["auction_key", "RUC_level"]].merge(audit_ann[["auction_key", gap_col]], on="auction_key")
    gap_rows = merged.loc[merged[gap_col] == True]
    assert len(gap_rows) > 0, "expected at least one real auction with a detected RUC source gap"
    assert gap_rows["RUC_level"].notna().all(), "a stale carry-forward value must still be a real number, never NaN"
    assert not (gap_rows["RUC_level"] == 0).any(), "a source gap must never be silently zero-filled"


# ---------------------------------------------------------------------------
# Mandatory timing enforcement (acceptance-review addition, Issue 4)
# ---------------------------------------------------------------------------


def test_assert_audit_table_availability_safe_passes_on_valid_data(joined, entries):
    from treasury_auction_stress.features.feature_matrix import (
        assert_audit_table_availability_safe,
    )

    for cutoff_name, cutoff_col in (("announcement", ANNOUNCEMENT_CUTOFF_COL), ("pre_auction", PRE_AUCTION_CUTOFF_COL)):
        matrix = build_predictor_matrix(cutoff_name=cutoff_name, joined=joined, entries=entries)
        audit = build_audit_table(cutoff_name=cutoff_name, joined=joined, entries=entries)
        results = assert_audit_table_availability_safe(audit, matrix, cutoff_col=cutoff_col)
        assert "dealer_stats" in results
        assert "treasury_rates" in results
        assert "cftc_positioning" in results
        assert any(k.startswith("rtdsm_") for k in results)
        for family, r in results.items():
            assert r["n_violations"] == 0, f"{family}: {r}"


def test_assert_audit_table_availability_safe_raises_on_a_single_violation(joined, entries):
    from treasury_auction_stress.features.feature_matrix import (
        assert_audit_table_availability_safe,
    )

    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    audit = build_audit_table(cutoff_name="announcement", joined=joined, entries=entries)

    poisoned = audit.copy()
    col = "dealer_stats__publication_safe_available_date"
    assert col in poisoned.columns
    # Push exactly one row's safe-availability date to far after its own cutoff.
    poisoned.loc[poisoned.index[0], col] = pd.Timestamp("2999-01-01")

    with pytest.raises(AssertionError, match="dealer_stats"):
        assert_audit_table_availability_safe(poisoned, matrix, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)


def test_assert_audit_table_availability_safe_raises_on_rtdsm_violation(joined, entries):
    from treasury_auction_stress.features.feature_matrix import (
        assert_audit_table_availability_safe,
    )

    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    audit = build_audit_table(cutoff_name="announcement", joined=joined, entries=entries)

    poisoned = audit.copy()
    lag_col = "rtdsm_macro__ROUTPUT_vintage_lag_calendar_days"
    matched_col = "rtdsm_macro__ROUTPUT_join_matched"
    assert lag_col in poisoned.columns and matched_col in poisoned.columns
    idx = poisoned.index[poisoned[matched_col].fillna(False).astype(bool)][0]
    poisoned.loc[idx, lag_col] = -5  # a negative lag means the vintage is after the cutoff

    with pytest.raises(AssertionError, match="rtdsm ROUTPUT"):
        assert_audit_table_availability_safe(poisoned, matrix, cutoff_col=ANNOUNCEMENT_CUTOFF_COL)


def test_validate_audit_table_docstring_matches_behavior_key_checks_only(joined, entries):
    """validate_audit_table must NOT catch a timing violation -- that is
    assert_audit_table_availability_safe's job. This test pins down the
    documented separation of concerns so the two are never conflated
    again.
    """
    matrix = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    audit = build_audit_table(cutoff_name="announcement", joined=joined, entries=entries)
    poisoned = audit.copy()
    poisoned.loc[poisoned.index[0], "dealer_stats__publication_safe_available_date"] = pd.Timestamp("2999-01-01")
    validate_audit_table(poisoned, matrix, cutoff_name="announcement")  # must NOT raise -- keys are still fine


# ---------------------------------------------------------------------------
# Reproducibility digest (acceptance-review addition, Issue 5)
# ---------------------------------------------------------------------------


def test_content_digest_is_deterministic():
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert compute_content_digest(df) == compute_content_digest(df.copy())


def test_content_digest_changes_when_a_value_is_mutated():
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    mutated = df.copy()
    mutated.loc[0, "a"] = 999
    assert compute_content_digest(df) != compute_content_digest(mutated)


def test_content_digest_changes_when_a_column_is_reordered():
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert compute_content_digest(df) != compute_content_digest(df[["b", "a"]])


def test_content_digest_changes_when_rows_are_swapped():
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    swapped = df.iloc[[1, 0, 2]].reset_index(drop=True)
    assert compute_content_digest(df) != compute_content_digest(swapped)


def test_content_digest_changes_when_a_dtype_changes():
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64")})
    as_float = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="float64")})
    assert compute_content_digest(df) != compute_content_digest(as_float)


def test_content_digest_is_order_sensitive_unlike_a_hash_sum():
    """The specific defect this replaces: pandas.util.hash_pandas_object(...).sum()
    is order-INsensitive (a sum), so two dataframes with the same rows
    in a different order produce the identical old-style fingerprint.
    The new digest must NOT have this property.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    reordered = df.iloc[[2, 1, 0]].reset_index(drop=True)

    old_style_fingerprint = int(pd.util.hash_pandas_object(df, index=False).sum())
    old_style_fingerprint_reordered = int(pd.util.hash_pandas_object(reordered, index=False).sum())
    assert old_style_fingerprint == old_style_fingerprint_reordered  # the old defect, reproduced

    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    assert compute_content_digest(df) != compute_content_digest(reordered)  # the fix


def test_content_digest_matches_across_two_full_pipeline_runs(joined, entries):
    """Run the full announcement-matrix build twice from the same
    synthetic inputs and confirm the digest matches -- a direct,
    minimal proxy for "run generation twice and confirm the digests
    match," exercised at full CLI scale in the real-data test below.
    """
    from treasury_auction_stress.features.feature_matrix import compute_content_digest

    first = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    second = build_predictor_matrix(cutoff_name="announcement", joined=joined, entries=entries)
    assert compute_content_digest(first) == compute_content_digest(second)


# ---------------------------------------------------------------------------
# Real-data regression (skipped if the local processed artifacts are absent)
# ---------------------------------------------------------------------------


def test_real_ordinary_modeling_sample_produces_expected_row_counts(entries):
    processed = Path("data/processed")
    required = [
        "treasury_auctions_nominal_coupons.parquet",
        "dealer_stats_wide.parquet",
        "treasury_rates_wide.parquet",
        "cftc_positioning_wide.parquet",
        "rtdsm_vintage_index.parquet",
        "rtdsm_snapshots.parquet",
    ]
    if not all((processed / f).exists() for f in required):
        pytest.skip("processed Phase 1/3/4 tables not present in this environment")

    from treasury_auction_stress.features.feature_matrix_cli import load_source_tables

    _nominal_df, tables = load_source_tables(processed)
    joined_real = build_join_tables(tables)
    ann = build_predictor_matrix(cutoff_name="announcement", joined=joined_real, entries=entries)
    pre = build_predictor_matrix(cutoff_name="pre_auction", joined=joined_real, entries=entries)
    validate_predictor_matrix(ann, entries, expected_rows=len(tables.sample))
    validate_predictor_matrix(pre, entries, expected_rows=len(tables.sample))
    assert len(ann) == len(tables.sample)

    targets = build_target_table(tables.sample)
    validate_target_table(targets, entries, expected_rows=len(tables.sample))

    availability = assert_safe_availability_never_exceeds_cutoff(joined_real)
    assert all(r["n_violations"] == 0 for r in availability.values())
    monotonicity = assert_cross_cutoff_monotonicity(joined_real)
    assert all(r["n_violations"] == 0 for r in monotonicity.values())

    from treasury_auction_stress.features.feature_matrix import (
        assert_audit_table_availability_safe,
    )

    for cutoff_name, cutoff_col, matrix in (
        ("announcement", ANNOUNCEMENT_CUTOFF_COL, ann),
        ("pre_auction", PRE_AUCTION_CUTOFF_COL, pre),
    ):
        audit = build_audit_table(cutoff_name=cutoff_name, joined=joined_real, entries=entries)
        validate_audit_table(audit, matrix, cutoff_name=cutoff_name)
        results = assert_audit_table_availability_safe(audit, matrix, cutoff_col=cutoff_col)
        assert all(r["n_violations"] == 0 for r in results.values()), results

    # Key-alignment proof on the real, full 1,279-row sample: shuffling
    # any one intermediate join table must not change the final output
    # (both cutoffs).
    rng = np.random.default_rng(7)
    baseline_by_cutoff = {"announcement": ann, "pre_auction": pre}
    for table_key in ("auction_structure", "dealer", "rates", "cftc", "rtdsm"):
        table = joined_real[table_key]
        shuffled = dict(joined_real)
        shuffled[table_key] = table.iloc[rng.permutation(len(table))].reset_index(drop=True)
        for cutoff_name, baseline in baseline_by_cutoff.items():
            result = build_predictor_matrix(cutoff_name=cutoff_name, joined=shuffled, entries=entries)
            pd.testing.assert_frame_equal(baseline, result)

    # Targeted, cheap regression for the real bug the full-scale checks
    # found (see cftc_join.py's module comment): shuffling the joined
    # tables above is NOT equivalent to this, because that bug lived
    # inside cftc_join.as_of_join's own merge_asof tie-breaking, which
    # depends on the RAW cftc_wide input's row order, not the joined
    # output's. The full-scale, all-5-source, both-cutoff version of
    # this check (per_source_input_order_invariance/
    # per_source_future_row_invariance) is exercised once, for real,
    # by tests/test_feature_matrix_cli.py's successful-run test (which
    # calls the actual CLI's mandatory gates) -- not duplicated here,
    # to avoid ~30s of redundant real-data computation in this file too.
    rng2 = np.random.default_rng(99)
    shuffled_cftc_wide = tables.cftc_wide.iloc[rng2.permutation(len(tables.cftc_wide))].reset_index(drop=True)
    shuffled_tables = replace(tables, cftc_wide=shuffled_cftc_wide)
    rejoined = build_join_tables(shuffled_tables)
    for cutoff_name, baseline in baseline_by_cutoff.items():
        result = build_predictor_matrix(cutoff_name=cutoff_name, joined=rejoined, entries=entries)
        pd.testing.assert_frame_equal(baseline, result)
