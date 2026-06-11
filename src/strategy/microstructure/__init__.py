"""Microstructure analytics primitives for the DVSLA strategy.

Pure, deterministic statistical building blocks (volume bars, rolling
statistics, order-flow imbalance, OI normalization, Hurst exponent) that are
shared between the live strategy engine and the backtest harness.
"""

from __future__ import annotations

from src.strategy.microstructure.flow_imbalance import (
    FlowImbalanceTracker,
    flow_imbalance,
    signed_volume,
    trade_sign,
)
from src.strategy.microstructure.hurst import HurstRegime, classify_hurst, hurst_rs
from src.strategy.microstructure.oi_zscore import OIZScoreTracker
from src.strategy.microstructure.rolling_stats import RollingStats, Welford, z_score
from src.strategy.microstructure.volume_bars import VolumeBar, VolumeBarAggregator

__all__ = [
    "FlowImbalanceTracker",
    "flow_imbalance",
    "signed_volume",
    "trade_sign",
    "HurstRegime",
    "classify_hurst",
    "hurst_rs",
    "OIZScoreTracker",
    "RollingStats",
    "Welford",
    "z_score",
    "VolumeBar",
    "VolumeBarAggregator",
]
