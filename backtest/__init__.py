"""Backtest harness for the DVSLA strategy.

This package records live market events to disk and replays them through the
same ``on_market_event`` interface used in production, so the strategy logic is
exercised identically in live and backtest modes (no look-ahead bias possible).

Names are resolved lazily (PEP 562). Importing them eagerly here would mean
that any use of any submodule drags in the whole trading stack, and one of the
callers is the nightly compaction cron on the bot's host, which runs against a
system interpreter with none of the bot's dependencies installed. Keeping
``backtest.recording_paths`` reachable without ``websockets`` is the point;
``from backtest import EventRecorder`` still works exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type checkers only, never at runtime
    from backtest.metrics import Metrics, compute_metrics
    from backtest.recorder import EventRecorder, serialize_event
    from backtest.replay import EventReplayer, deserialize_event, replay_file
    from backtest.simulator import (
        BacktestResult,
        BacktestSimulator,
        ClosedTrade,
        OpenPosition,
        SimConfig,
    )
    from backtest.sweep import SweepResult, default_grid, sweep

_EXPORTS = {
    "EventRecorder": "backtest.recorder",
    "serialize_event": "backtest.recorder",
    "EventReplayer": "backtest.replay",
    "deserialize_event": "backtest.replay",
    "replay_file": "backtest.replay",
    "Metrics": "backtest.metrics",
    "compute_metrics": "backtest.metrics",
    "BacktestResult": "backtest.simulator",
    "BacktestSimulator": "backtest.simulator",
    "ClosedTrade": "backtest.simulator",
    "OpenPosition": "backtest.simulator",
    "SimConfig": "backtest.simulator",
    "SweepResult": "backtest.sweep",
    "default_grid": "backtest.sweep",
    "sweep": "backtest.sweep",
}

__all__ = [*_EXPORTS]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
