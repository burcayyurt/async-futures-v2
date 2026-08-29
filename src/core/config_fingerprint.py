"""Stable identity for the settings that determine how a trade turns out.

The observation plan is "change one thing, watch, repeat", which only works if
the journal can be sliced by configuration afterwards. A trade row that does not
say which parameters produced it is unattributable the moment anything changes,
so every closed trade carries a short ``config_id`` and the full snapshot is
written once to a registry file.

Only fields that can change a trade's outcome are fingerprinted. Paths, tokens
and logging levels are excluded so that cosmetic edits do not split the history
into artificially separate configurations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.config import HyperliquidSettings

logger = logging.getLogger(__name__)

# Ordered for readability of the snapshot; the hash sorts keys itself.
FINGERPRINTED_FIELDS: tuple[str, ...] = (
    # Universe and sizing
    "symbols",
    "leverage",
    "margin_mode",
    "trade_risk_pct",
    "confidence_sizing_enabled",
    "confidence_size_floor",
    # Entry mechanics
    "maker_entry_enabled",
    "maker_entry_offset_bps",
    "reentry_cooldown_seconds",
    "ws_reconnect_entry_cooldown_seconds",
    # Exit mechanics
    "trailing_callback_pct",
    "break_even_trigger_pct",
    "atr_stop_enabled",
    "atr_window",
    "atr_stop_mult",
    "atr_stop_min_pct",
    "reversion_tp_enabled",
    "reversion_window",
    "min_hold_seconds",
    "max_hold_seconds",
    "mark_staleness_gap_seconds",
    # Costs
    "maker_fee_bps",
    "taker_fee_bps",
    # Strategy engine
    "strategy_engine",
    "dvsla_invert",
    "dvsla_ret_z_window",
    "dvsla_ret_z_entry",
    "dvsla_ret_z_clamp",
    "dvsla_ret_z_reject",
    "dvsla_ret_min_abs_pct",
    "dvsla_flow_imbalance_min",
    "dvsla_oi_z_window",
    "dvsla_oi_z_drop",
    "dvsla_fade_oi_z_veto",
    "dvsla_hurst_window",
    "dvsla_hurst_max",
    "dvsla_warmup_bars",
    "dvsla_cooldown_bars",
    "dvsla_min_confidence",
    # Risk guards
    "correlation_guard_enabled",
    "correlation_max_same_side",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Normalize so 0.35 and 0.3500 fingerprint identically.
        return str(value.normalize())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_snapshot(settings: HyperliquidSettings) -> dict[str, Any]:
    """The outcome-relevant settings, as plain JSON-safe values."""

    snapshot: dict[str, Any] = {}
    for field in FINGERPRINTED_FIELDS:
        if not hasattr(settings, field):
            # Field removed or renamed: skip rather than crash the bot on startup.
            continue
        snapshot[field] = _jsonable(getattr(settings, field))
    return snapshot


def config_id(settings: HyperliquidSettings) -> str:
    """Short stable id for a configuration.

    Sorted-key JSON keeps the digest independent of field order, and 12 hex chars
    is far beyond collision risk for the handful of configurations a research
    log will ever hold, while staying readable in a JSONL row.
    """

    payload = json.dumps(config_snapshot(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def register_config(settings: HyperliquidSettings, path: Path | str) -> str:
    """Append this configuration to the registry unless already recorded.

    The registry is what turns a ``config_id`` in a trade row back into readable
    parameters months later. Appending only new ids keeps one line per distinct
    configuration and preserves the order they were first used in.
    """

    identifier = config_id(settings)
    registry = Path(path)

    if registry.exists():
        try:
            with registry.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if json.loads(line).get("config_id") == identifier:
                            return identifier
                    except json.JSONDecodeError:
                        continue
        except OSError:
            logger.warning("Could not read config registry %s", registry, exc_info=True)

    entry = {
        "config_id": identifier,
        "first_seen": datetime.now(timezone.utc).isoformat(),
        "settings": config_snapshot(settings),
    }
    try:
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Registered new configuration %s in %s", identifier, registry)
    except OSError:
        logger.warning("Could not write config registry %s", registry, exc_info=True)
    return identifier
