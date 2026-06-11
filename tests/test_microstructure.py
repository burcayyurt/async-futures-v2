"""Unit tests for the DVSLA microstructure primitives."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.strategy.microstructure.flow_imbalance import (
    FlowImbalanceTracker,
    flow_imbalance,
    signed_volume,
    trade_sign,
)
from src.strategy.microstructure.hurst import HurstRegime, classify_hurst, hurst_rs
from src.strategy.microstructure.oi_zscore import OIZScoreTracker
from src.strategy.microstructure.rolling_stats import RollingStats, Welford, z_score
from src.strategy.microstructure.volume_bars import VolumeBarAggregator


def _ts(seconds: float = 0.0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)


# --------------------------------------------------------------------------- #
# rolling_stats
# --------------------------------------------------------------------------- #
def test_z_score_degenerate_std_returns_zero() -> None:
    assert z_score(5.0, 3.0, 0.0) == 0.0
    assert z_score(5.0, 5.0, 2.0) == 0.0


def test_welford_matches_statistics_module() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    w = Welford()
    w.extend(values)
    assert w.count == len(values)
    assert w.mean == pytest.approx(statistics.fmean(values))
    assert w.std == pytest.approx(statistics.stdev(values))


def test_welford_single_value_variance_zero() -> None:
    w = Welford()
    w.update(42.0)
    assert w.variance == 0.0
    assert w.std == 0.0


def test_rolling_stats_windows_and_evicts() -> None:
    rs = RollingStats(window=3)
    rs.extend([1.0, 2.0, 3.0])
    assert rs.is_full
    assert rs.mean == pytest.approx(2.0)
    # Adding 4.0 evicts 1.0 -> window is [2,3,4]
    rs.update(4.0)
    assert rs.count == 3
    assert rs.mean == pytest.approx(3.0)
    assert rs.std == pytest.approx(statistics.stdev([2.0, 3.0, 4.0]))


def test_rolling_stats_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        RollingStats(window=0)


# --------------------------------------------------------------------------- #
# flow_imbalance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "side,expected",
    [
        ("B", 1),
        ("buy", 1),
        ("Bid", 1),
        ("A", -1),
        ("sell", -1),
        ("ASK", -1),
        ("", 0),
        ("???", 0),
    ],
)
def test_trade_sign(side: str, expected: int) -> None:
    assert trade_sign(side) == expected


def test_signed_volume_sign_and_type() -> None:
    assert signed_volume("B", Decimal("2.5")) == Decimal("2.5")
    assert signed_volume("A", Decimal("2.5")) == Decimal("-2.5")
    assert signed_volume("x", Decimal("2.5")) == Decimal("0")


def test_flow_imbalance_bounds() -> None:
    assert flow_imbalance(0.0, 0.0) == 0.0
    assert flow_imbalance(10.0, 0.0) == 1.0
    assert flow_imbalance(0.0, 10.0) == -1.0
    assert flow_imbalance(3.0, 1.0) == pytest.approx(0.5)


def test_flow_imbalance_tracker_rolls_window() -> None:
    tracker = FlowImbalanceTracker(window=2)
    tracker.update("B", Decimal("1"))  # buys=1
    tracker.update("B", Decimal("1"))  # buys=2
    assert tracker.imbalance == pytest.approx(1.0)
    # Evict oldest buy, add a sell -> [buy1, sell1]
    tracker.update("A", Decimal("1"))
    assert tracker.count == 2
    assert tracker.imbalance == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# volume_bars
# --------------------------------------------------------------------------- #
def test_volume_bar_closes_on_threshold() -> None:
    agg = VolumeBarAggregator("BTC", threshold=Decimal("10"))
    bars = agg.update(Decimal("100"), Decimal("4"), "B", _ts(0))
    assert bars == []
    bars = agg.update(Decimal("102"), Decimal("6"), "B", _ts(1))
    assert len(bars) == 1
    bar = bars[0]
    assert bar.volume == Decimal("10")
    assert bar.open_px == Decimal("100")
    assert bar.close_px == Decimal("102")
    assert bar.high_px == Decimal("102")
    assert bar.signed_volume == Decimal("10")
    assert bar.imbalance == pytest.approx(1.0)


def test_volume_bar_splits_large_trade() -> None:
    agg = VolumeBarAggregator("ETH", threshold=Decimal("5"))
    # A single size-12 trade should close two full bars (5+5) and leave 2.
    bars = agg.update(Decimal("50"), Decimal("12"), "A", _ts(0))
    assert len(bars) == 2
    assert all(b.volume == Decimal("5") for b in bars)
    assert all(b.signed_volume == Decimal("-5") for b in bars)
    assert agg.has_partial  # 2 units remain in progress


def test_volume_bar_high_low_tracking() -> None:
    agg = VolumeBarAggregator("SOL", threshold=Decimal("3"))
    agg.update(Decimal("100"), Decimal("1"), "B", _ts(0))
    agg.update(Decimal("110"), Decimal("1"), "A", _ts(1))
    bars = agg.update(Decimal("90"), Decimal("1"), "B", _ts(2))
    assert len(bars) == 1
    bar = bars[0]
    assert bar.high_px == Decimal("110")
    assert bar.low_px == Decimal("90")
    assert bar.close_px == Decimal("90")
    assert bar.ret == pytest.approx(float((Decimal("90") - Decimal("100")) / Decimal("100")))


def test_volume_bar_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        VolumeBarAggregator("BTC", threshold=0)


# --------------------------------------------------------------------------- #
# oi_zscore
# --------------------------------------------------------------------------- #
def test_oi_zscore_first_observation_zero() -> None:
    tracker = OIZScoreTracker(window=10)
    assert tracker.update(Decimal("1000")) == 0.0
    assert not tracker.is_ready


def test_oi_zscore_detects_anomalous_drop() -> None:
    tracker = OIZScoreTracker(window=20)
    base = Decimal("1000")
    # Feed small noisy positive increments to build a non-degenerate
    # distribution of OI deltas (alternating ~ +5 / +15).
    prev = base
    for i in range(1, 13):
        step = Decimal("5") if i % 2 == 0 else Decimal("15")
        prev = prev + step
        tracker.update(prev)
    # A large negative jump (liquidation cascade) should yield a strongly
    # negative z-score versus the recent small-positive deltas.
    z = tracker.update(prev - Decimal("500"))
    assert z < -2.0


def test_oi_zscore_skips_unchanged_readings() -> None:
    # Sparse feeds (lower-volume coins) repeat the same OI ~90% of the time.
    # Those zero-deltas must not enter the window, else a later real move's
    # z-score blows up. Carry the last meaningful z forward instead.
    tracker = OIZScoreTracker(window=20)
    oi = Decimal("1000000")
    for _ in range(50):
        assert tracker.update(oi) == 0.0  # repeated -> skipped, no window pollution
    assert tracker.count == 0
    # A handful of genuine small moves builds a sane distribution.
    for delta in (Decimal("50"), Decimal("-30"), Decimal("40"), Decimal("-20"), Decimal("60")):
        oi += delta
        tracker.update(oi)
    assert tracker.count == 5


def test_oi_zscore_clamps_extreme_values() -> None:
    # Without the clamp, a near-degenerate window produced z > 1e6 in practice.
    tracker = OIZScoreTracker(window=20, z_clamp=10.0)
    oi = Decimal("1000000")
    for delta in (Decimal("50"), Decimal("-30"), Decimal("40"), Decimal("-20"), Decimal("60")):
        oi += delta
        tracker.update(oi)
    z = tracker.update(oi - Decimal("500000"))  # massive collapse
    assert z == -10.0


def test_oi_zscore_can_disable_skip_unchanged() -> None:
    tracker = OIZScoreTracker(window=20, skip_unchanged=False)
    tracker.update(Decimal("1000"))
    tracker.update(Decimal("1000"))  # zero delta fed into window
    assert tracker.count == 1


# --------------------------------------------------------------------------- #
# hurst
# --------------------------------------------------------------------------- #
def test_hurst_short_series_returns_half() -> None:
    assert hurst_rs([1.0, 2.0, 3.0]) == 0.5


def test_hurst_trending_series_above_half() -> None:
    # A cumulative random walk with drift is persistent (H > 0.5).
    rng = np.random.default_rng(7)
    steps = rng.normal(0.5, 1.0, size=512)
    series = np.cumsum(steps)
    h = hurst_rs(series)
    assert h > 0.55


def test_hurst_mean_reverting_series_below_half() -> None:
    # Strongly anti-correlated series (alternating) is mean-reverting.
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.1, size=512)
    series = np.array([(-1.0) ** i for i in range(512)]) + noise
    h = hurst_rs(series)
    assert h < 0.45


def test_classify_hurst_buckets() -> None:
    assert classify_hurst(0.3) is HurstRegime.MEAN_REVERTING
    assert classify_hurst(0.5) is HurstRegime.RANDOM_WALK
    assert classify_hurst(0.8) is HurstRegime.TRENDING
