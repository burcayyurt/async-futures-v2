"""Tests for the correlation-trap market breadth guard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.strategy.market_breadth import DOWN, UP, MarketBreadthTracker

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    return BASE_TS + timedelta(seconds=seconds)


def test_counts_distinct_same_side_coins() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=4)
    for i, coin in enumerate(["ADA", "SOL", "BNB"]):
        tracker.record(coin, DOWN, _ts(i))

    assert tracker.count_same_side(DOWN, _ts(3)) == 3
    assert tracker.count_same_side(UP, _ts(3)) == 0


def test_duplicate_coin_counts_once() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=4)
    tracker.record("FET", DOWN, _ts(0))
    tracker.record("FET", DOWN, _ts(1))
    tracker.record("FET", DOWN, _ts(2))

    assert tracker.count_same_side(DOWN, _ts(3)) == 1


def test_exclude_self_is_not_counted() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=4)
    tracker.record("FET", DOWN, _ts(0))
    tracker.record("ADA", DOWN, _ts(1))

    assert tracker.count_same_side(DOWN, _ts(2), exclude="FET") == 1


def test_is_synchronized_triggers_at_threshold() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=4)
    for i, coin in enumerate(["ADA", "SOL", "BNB", "BTC", "TIA", "SEI"]):
        tracker.record(coin, DOWN, _ts(i))

    # FET (the 7th down-move) sees 6 other coins moving down -> blocked.
    assert tracker.is_synchronized(DOWN, _ts(7), exclude="FET") is True
    # The opposite direction is unaffected.
    assert tracker.is_synchronized(UP, _ts(7), exclude="FET") is False


def test_below_threshold_not_synchronized() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=4)
    for i, coin in enumerate(["ADA", "SOL", "BNB"]):
        tracker.record(coin, DOWN, _ts(i))

    # 3 other coins, threshold is 4 -> not yet synchronized.
    assert tracker.is_synchronized(DOWN, _ts(4), exclude="XRP") is False


def test_stale_marks_expire_outside_window() -> None:
    tracker = MarketBreadthTracker(window_seconds=60, max_same_side=2)
    tracker.record("ADA", DOWN, _ts(0))
    tracker.record("SOL", DOWN, _ts(10))

    # 120s later both records are stale.
    assert tracker.count_same_side(DOWN, _ts(120)) == 0
    assert tracker.is_synchronized(DOWN, _ts(120)) is False


def test_guard_disabled_when_max_is_zero() -> None:
    tracker = MarketBreadthTracker(window_seconds=120, max_same_side=0)
    for i, coin in enumerate(["ADA", "SOL", "BNB", "BTC", "TIA"]):
        tracker.record(coin, DOWN, _ts(i))

    assert tracker.is_synchronized(DOWN, _ts(6)) is False
