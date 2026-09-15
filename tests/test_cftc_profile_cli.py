from __future__ import annotations

import pandas as pd
import pytest

from treasury_auction_stress.features.cftc_join import (
    DIRECT_CONTRACT_STATUS_AVAILABLE,
    DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE,
    DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR,
)
from treasury_auction_stress.features.cftc_profile_cli import (
    RECONCILIATION_DIRECT_AVAILABLE,
    RECONCILIATION_DIRECT_NO_DATA,
    RECONCILIATION_NO_DIRECT_CONTRACT,
    RECONCILIATION_NO_REPORT,
    row_level_reconciliation,
)


def _joined(rows):
    return pd.DataFrame(rows)


def test_reconciliation_states_sum_exactly_to_row_count():
    joined = _joined(
        [
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_AVAILABLE},
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR},
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE},
            {"cftc_report_available": False, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR},
        ]
    )
    result = row_level_reconciliation(joined)
    assert result["n_auctions"].sum() == 4


def test_reconciliation_assigns_each_state_correctly():
    joined = _joined(
        [
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_AVAILABLE},
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR},
            {"cftc_report_available": True, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_MAPPED_BUT_DATA_UNAVAILABLE},
            {"cftc_report_available": False, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_NO_CONTRACT_FOR_TENOR},
        ]
    )
    result = row_level_reconciliation(joined).set_index("state")["n_auctions"]
    assert result[RECONCILIATION_DIRECT_AVAILABLE] == 1
    assert result[RECONCILIATION_NO_DIRECT_CONTRACT] == 1
    assert result[RECONCILIATION_DIRECT_NO_DATA] == 1
    assert result[RECONCILIATION_NO_REPORT] == 1


def test_reconciliation_states_are_mutually_exclusive():
    """A row with cftc_report_available=False must land in exactly the
    'no report' state, regardless of what its direct_contract_mapping_status
    happens to be (it must never double-count into a second state).
    """
    joined = _joined(
        [{"cftc_report_available": False, "direct_contract_mapping_status": DIRECT_CONTRACT_STATUS_AVAILABLE}]
    )
    result = row_level_reconciliation(joined).set_index("state")["n_auctions"]
    assert result[RECONCILIATION_NO_REPORT] == 1
    assert len(result) == 1


def test_real_ordinary_modeling_sample_reconciliation_sums_to_1279():
    """Regression test against this project's own live-generated
    processed tables, when available -- skipped otherwise (this is a
    repository-state check, not a network call).
    """
    from pathlib import Path

    processed = Path("data/processed")
    if not (processed / "cftc_positioning_wide.parquet").exists():
        pytest.skip("processed CFTC tables not present in this environment")

    from treasury_auction_stress.features.auction_cutoffs import (
        ANNOUNCEMENT_CUTOFF_COL,
        add_cutoff_dates,
    )
    from treasury_auction_stress.features.cftc_join import (
        add_tenor_matched_positioning_features,
        as_of_join,
    )
    from treasury_auction_stress.features.eligibility import (
        select_analysis_sample,
        select_modeling_sample,
    )

    nominal_df = pd.read_parquet(processed / "treasury_auctions_nominal_coupons.parquet")
    wide_df = pd.read_parquet(processed / "cftc_positioning_wide.parquet")
    settled, _pending = select_analysis_sample(nominal_df)
    ordinary_modeling_sample = add_cutoff_dates(select_modeling_sample(settled))
    joined = add_tenor_matched_positioning_features(as_of_join(ordinary_modeling_sample, wide_df, cutoff_col=ANNOUNCEMENT_CUTOFF_COL))
    result = row_level_reconciliation(joined)
    assert result["n_auctions"].sum() == len(ordinary_modeling_sample) == 1279
