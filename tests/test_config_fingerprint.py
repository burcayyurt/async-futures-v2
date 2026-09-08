"""The fingerprint has to cover everything that changes an outcome.

A field left out of ``FINGERPRINTED_FIELDS`` does not fail loudly: two runs with
materially different settings keep the same ``config_id``, their trades pool
into one series, and the mistake only surfaces when someone asks why a result
will not reproduce. The volume-bar thresholds were missing until 2026-09-08,
which would have merged the pre- and post-rescale arms.
"""

from __future__ import annotations

from decimal import Decimal

from src.core.config import HyperliquidSettings
from src.core.config_fingerprint import (
    FINGERPRINTED_FIELDS,
    config_id,
    config_snapshot,
)


def _settings(thresholds: dict | None = None) -> HyperliquidSettings:
    """Build settings, overriding the thresholds by their validation alias.

    ``populate_by_name`` is not enabled, so passing ``dvsla_symbol_thresholds``
    by field name is silently ignored and the defaults come back instead --
    which makes a test look like it passed against an unchanged config.
    """

    if thresholds is None:
        return HyperliquidSettings()
    return HyperliquidSettings(DVSLA_SYMBOL_THRESHOLDS=thresholds)


def test_bar_definition_is_fingerprinted() -> None:
    """The bar thresholds decide what every later statistic is computed on."""

    assert "dvsla_symbol_thresholds" in FINGERPRINTED_FIELDS
    assert "dvsla_volume_bar_threshold" in FINGERPRINTED_FIELDS


def test_changing_a_symbol_threshold_changes_the_id() -> None:
    base = _settings()
    moved = dict(base.dvsla_symbol_thresholds)
    moved["DOT"] = moved["DOT"] * 2

    assert config_id(base) != config_id(_settings(moved))


def test_threshold_dict_order_does_not_change_the_id() -> None:
    """Defaults and a DVSLA_SYMBOL_THRESHOLDS env var arrive in different orders.

    Identical settings must fingerprint identically regardless of how they were
    loaded, or every restart from a different source looks like a new arm.
    """

    base = _settings()
    reversed_order = dict(reversed(list(base.dvsla_symbol_thresholds.items())))

    # Dicts compare equal regardless of order, so assert on the key sequence.
    assert list(reversed_order) != list(base.dvsla_symbol_thresholds)
    assert config_id(_settings(reversed_order)) == config_id(base)


def test_equivalent_decimals_fingerprint_identically() -> None:
    base = _settings()
    padded = dict(base.dvsla_symbol_thresholds)
    padded["BTC"] = Decimal("8.000")

    assert config_id(_settings(padded)) == config_id(base)


def test_snapshot_covers_every_declared_field() -> None:
    snapshot = config_snapshot(_settings())
    settings = _settings()

    for field in FINGERPRINTED_FIELDS:
        if hasattr(settings, field):
            assert field in snapshot, f"{field} declared but missing from snapshot"
