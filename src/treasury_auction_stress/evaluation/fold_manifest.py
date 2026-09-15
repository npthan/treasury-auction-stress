"""Phase 6: the fold manifest -- one row per (cutoff view, fold),
identifying the exact train/test key sets (a deterministic hash of the
sorted training keys, plus the sorted training keys themselves for
small folds) so every reported result can be reproduced without
re-running the whole pipeline. Written to
`data/processed/phase_6_fold_manifest.parquet` (gitignored, per
`docs/project_rules.md`'s "final analysis-ready tables ... gitignored").
"""

from __future__ import annotations

import hashlib

import pandas as pd

from treasury_auction_stress.evaluation.timing import build_fold
from treasury_auction_stress.features.feature_matrix import AUCTION_KEY_COL


def _key_hash(keys: list[str]) -> str:
    canonical = "\x1f".join(sorted(keys))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_fold_manifest(
    *, frame_by_cutoff: dict[str, pd.DataFrame], fold_years_with_provisional_flag: list[tuple[int, bool]]
) -> pd.DataFrame:
    rows = []
    for cutoff_view, frame in frame_by_cutoff.items():
        for fold_year, is_provisional in fold_years_with_provisional_flag:
            fold = build_fold(frame, test_year=fold_year)
            train_keys = sorted(fold.train[AUCTION_KEY_COL].tolist())
            test_keys = sorted(fold.test[AUCTION_KEY_COL].tolist())
            rows.append(
                {
                    "cutoff_view": cutoff_view,
                    "fold_id": f"{fold_year}_provisional" if is_provisional else str(fold_year),
                    "test_year": fold_year,
                    "is_provisional": is_provisional,
                    "fit_origin": fold.fit_origin,
                    "n_train": len(train_keys),
                    "n_test": len(test_keys),
                    "n_prior_year_rows_excluded_by_availability_rule": fold.n_train_dropped_for_year_boundary,
                    "train_key_hash_sha256": _key_hash(train_keys),
                    "test_key_hash_sha256": _key_hash(test_keys),
                    "train_keys": ";".join(train_keys),
                    "test_keys": ";".join(test_keys),
                }
            )
    return pd.DataFrame(rows)
