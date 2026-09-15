from __future__ import annotations

import pandas as pd

from treasury_auction_stress.evaluation.fold_manifest import build_fold_manifest
from treasury_auction_stress.evaluation.timing import add_result_safe_available_date
from treasury_auction_stress.features.auction_cutoffs import ANNOUNCEMENT_CUTOFF_COL
from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL


def _pool() -> pd.DataFrame:
    rows = []
    for year in range(2010, 2016):
        for month in (1, 4, 7, 10):
            cusip = f"A{year}{month}"
            auction_date = pd.Timestamp(f"{year}-{month:02d}-15")
            rows.append(
                {
                    "cusip": cusip,
                    "tenor": "10-Year",
                    "auction_date": auction_date,
                    "announcemt_date": auction_date - pd.Timedelta(days=7),
                    ANNOUNCEMENT_CUTOFF_COL: auction_date - pd.Timedelta(days=7),
                    AUCTION_KEY_COL: f"{cusip}_{auction_date.date().isoformat()}",
                }
            )
    return add_result_safe_available_date(pd.DataFrame(rows))


def test_build_fold_manifest_basic():
    pool = _pool()
    manifest = build_fold_manifest(
        frame_by_cutoff={"announcement": pool, "pre_auction": pool},
        fold_years_with_provisional_flag=[(2015, False)],
    )
    assert len(manifest) == 2  # one per cutoff view
    row = manifest.iloc[0]
    assert row["n_test"] == 4
    assert row["n_train"] == 20
    assert len(row["train_keys"].split(";")) == row["n_train"]
    assert len(row["test_keys"].split(";")) == row["n_test"]


def test_fold_manifest_hashes_are_deterministic_and_order_independent():
    pool = _pool()
    shuffled = pool.sample(frac=1.0, random_state=3).reset_index(drop=True)
    manifest_a = build_fold_manifest(
        frame_by_cutoff={"announcement": pool}, fold_years_with_provisional_flag=[(2015, False)]
    )
    manifest_b = build_fold_manifest(
        frame_by_cutoff={"announcement": shuffled}, fold_years_with_provisional_flag=[(2015, False)]
    )
    assert manifest_a.iloc[0]["train_key_hash_sha256"] == manifest_b.iloc[0]["train_key_hash_sha256"]
    assert manifest_a.iloc[0]["test_key_hash_sha256"] == manifest_b.iloc[0]["test_key_hash_sha256"]
