"""Regression guard for configs/phase_9_release.yml: this file is the
machine-readable Phase 9 acceptance criteria, but nothing had ever
actually parsed it as YAML before the Phase 9 acceptance review -- it
contained two real syntax defects (an unquoted colon inside a list item,
and a mapping key placed directly under a block sequence) that made it
invalid YAML. Both were fixed as pure syntax corrections (no criterion's
meaning changed); this test prevents the file from silently regressing
back to unparseable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "phase_9_release.yml"


def test_phase_9_release_yaml_parses():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert "phase_9a_tree_and_history_audit" in config
    assert "phase_9_addendum_acceptance_review" in config
