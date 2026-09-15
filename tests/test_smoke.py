"""Environment and package smoke tests.

These are not tests of research logic. They exist so that a broken
environment (wrong Python version, missing dependency, unimportable
package) fails loudly and immediately, instead of surfacing later as a
confusing error deep in a data pipeline.
"""

import sys

import treasury_auction_stress


def test_python_version_is_pinned_range() -> None:
    """The project requires Python >=3.12,<3.13 (see pyproject.toml)."""
    assert sys.version_info[:2] == (3, 12)


def test_package_importable() -> None:
    """The reusable package under src/treasury_auction_stress must import."""
    assert hasattr(treasury_auction_stress, "main")


def test_core_dependencies_importable() -> None:
    """Every Phase 0 runtime dependency must actually be importable,
    not just listed in pyproject.toml.
    """
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import pydantic  # noqa: F401
    import requests  # noqa: F401
    import scipy  # noqa: F401
    import seaborn  # noqa: F401
    import sklearn  # noqa: F401
    import statsmodels  # noqa: F401
    import yaml  # noqa: F401
