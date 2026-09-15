"""Phase 7: `configs/phase_7_protocol.yml` loads cleanly and its
structural fold schedule matches Phase 6's frozen schedule exactly --
Phase 7 must never derive a different fold calendar.
"""

from __future__ import annotations

from pathlib import Path

from treasury_auction_stress.evaluation.protocol import load_protocol
from treasury_auction_stress.evaluation.protocol7 import load_protocol7


def test_phase7_protocol_loads():
    protocol7 = load_protocol7()
    assert protocol7.primary_target == "primary_dealer_share"
    assert protocol7.provisional_test_year == 2026
    assert len(protocol7.complete_test_years) == 11
    assert protocol7.quantile_levels == (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def test_phase7_fold_schedule_matches_phase6_exactly():
    protocol6 = load_protocol()
    protocol7 = load_protocol7()
    assert protocol7.complete_test_years == protocol6.complete_test_years
    assert protocol7.provisional_test_year == protocol6.provisional_test_year
    assert protocol7.initial_training_start == protocol6.initial_training_start
    assert protocol7.initial_training_end == protocol6.initial_training_end


def test_phase7_protocol_freeze_metadata_is_present():
    protocol7 = load_protocol7()
    assert protocol7.raw["metadata"]["frozen_before_scoring"] is True
    assert len(protocol7.raw["metadata"]["change_log"]) >= 1


def test_phase7_protocol_file_exists_and_is_readable_yaml():
    path = Path("configs/phase_7_protocol.yml")
    assert path.exists()
    protocol7 = load_protocol7(path)
    assert protocol7.raw is not None
