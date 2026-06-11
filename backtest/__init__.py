"""Backtest harness for the DVSLA strategy.

This package records live market events to disk and replays them through the
same ``on_market_event`` interface used in production, so the strategy logic is
exercised identically in live and backtest modes (no look-ahead bias possible).
"""

from __future__ import annotations

from backtest.recorder import EventRecorder, serialize_event
from backtest.replay import EventReplayer, deserialize_event, replay_file
from backtest.metrics import Metrics, compute_metrics
from backtest.simulator import (
    BacktestResult,
    BacktestSimulator,
    ClosedTrade,
    OpenPosition,
    SimConfig,
)
from backtest.sweep import SweepResult, default_grid, sweep

__all__ = [
    "EventRecorder",
    "serialize_event",
    "EventReplayer",
    "deserialize_event",
    "replay_file",
    "Metrics",
    "compute_metrics",
    "BacktestResult",
    "BacktestSimulator",
    "ClosedTrade",
    "OpenPosition",
    "SimConfig",
    "SweepResult",
    "default_grid",
    "sweep",
]
