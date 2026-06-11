"""Tests for the DVSLA backtest harness (simulator, metrics, sweep)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from backtest.metrics import compute_metrics
from backtest.simulator import (
    BacktestSimulator,
    ClosedTrade,
    SimConfig,
)
from backtest.sweep import default_grid, sweep
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.signals import SignalSide, TradeSignal

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(step: int) -> datetime:
    return BASE_TS + timedelta(seconds=step)


def _trade(coin: str, px: float, sz: float, side: str, step: int) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.TRADE,
        coin=coin,
        ts=_ts(step),
        payload=TradePayload(
            px=Decimal(str(px)), sz=Decimal(str(sz)), side=side, hash="0x", tid=None
        ),
    )


class _ScriptedStrategy:
    """Emits a preset signal when it sees a TRADE at a tagged step."""

    def __init__(self, signal_at: dict[int, TradeSignal]) -> None:
        self._signal_at = signal_at
        self._step = 0

    async def on_market_event(self, event: MarketEvent):
        produced = self._signal_at.get(self._step)
        self._step += 1
        return produced


def _closed(
    net: float,
    *,
    side: SignalSide = SignalSide.LONG,
    reason: str = "take_profit",
    notional: float = 1000.0,
    r: float = 1.0,
) -> ClosedTrade:
    return ClosedTrade(
        coin="BTC",
        side=side,
        entry_px=Decimal("100"),
        exit_px=Decimal("101"),
        qty=Decimal("10"),
        notional=Decimal(str(notional)),
        entry_ts=BASE_TS,
        exit_ts=BASE_TS,
        gross_pnl=Decimal(str(net)),
        fees=Decimal("0"),
        net_pnl=Decimal(str(net)),
        r_multiple=Decimal(str(r)),
        exit_reason=reason,
        confidence=Decimal("0.5"),
    )


# ----------------------------------------------------------------- simulator


async def test_long_take_profit_books_winner():
    coin = "BTC"
    signal = TradeSignal(
        symbol=coin,
        side=SignalSide.LONG,
        entry_mark_price=Decimal("100"),
        confidence=Decimal("0.5"),
        timestamp=_ts(0),
        reason="test",
    )
    # Signal fires on event index 0; price then rallies through the TP.
    strat = _ScriptedStrategy({0: signal})
    cfg = SimConfig(
        take_profit_pct=Decimal("0.01"),
        stop_loss_pct=Decimal("0.01"),
        time_stop_seconds=Decimal("0"),
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    events = [
        _trade(coin, 100, 1, "B", 0),
        _trade(coin, 100.5, 1, "B", 1),
        _trade(coin, 101.2, 1, "B", 2),  # crosses TP at 101
    ]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.net_pnl > 0


async def test_long_stop_books_loser():
    coin = "ETH"
    signal = TradeSignal(
        symbol=coin,
        side=SignalSide.LONG,
        entry_mark_price=Decimal("100"),
        confidence=Decimal("1"),
        timestamp=_ts(0),
        reason="test",
    )
    strat = _ScriptedStrategy({0: signal})
    cfg = SimConfig(
        take_profit_pct=Decimal("0.02"),
        stop_loss_pct=Decimal("0.01"),
        time_stop_seconds=Decimal("0"),
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    events = [
        _trade(coin, 100, 1, "A", 0),
        _trade(coin, 98.5, 1, "A", 1),  # crosses stop at 99
    ]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].net_pnl < 0


async def test_time_stop_closes_position():
    coin = "SOL"
    signal = TradeSignal(
        symbol=coin,
        side=SignalSide.LONG,
        entry_mark_price=Decimal("100"),
        confidence=Decimal("0.5"),
        timestamp=_ts(0),
        reason="test",
    )
    strat = _ScriptedStrategy({0: signal})
    cfg = SimConfig(
        take_profit_pct=Decimal("0.5"),
        stop_loss_pct=Decimal("0.5"),
        time_stop_seconds=Decimal("10"),
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    events = [
        _trade(coin, 100, 1, "B", 0),
        _trade(coin, 100.1, 1, "B", 5),
        _trade(coin, 100.2, 1, "B", 20),  # 20s elapsed > time stop
    ]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "time_stop"


async def test_one_position_per_symbol():
    coin = "BTC"

    def mk(step: int) -> TradeSignal:
        return TradeSignal(
            symbol=coin,
            side=SignalSide.LONG,
            entry_mark_price=Decimal("100"),
            confidence=Decimal("0.5"),
            timestamp=_ts(step),
            reason="test",
        )

    strat = _ScriptedStrategy({0: mk(0), 1: mk(1)})
    cfg = SimConfig(
        take_profit_pct=Decimal("0.5"),
        stop_loss_pct=Decimal("0.5"),
        time_stop_seconds=Decimal("0"),
    )
    events = [_trade(coin, 100, 1, "B", 0), _trade(coin, 100, 1, "B", 1)]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    # Second signal is skipped while the first is still open.
    assert result.signals == 2
    assert result.skipped_signals == 1
    assert len(result.trades) == 1  # closed at eod_mark


async def test_confidence_scales_notional():
    coin = "BTC"
    low = TradeSignal(
        symbol=coin, side=SignalSide.LONG, entry_mark_price=Decimal("100"),
        confidence=Decimal("0"), timestamp=_ts(0), reason="t",
    )
    cfg = SimConfig(
        min_notional=Decimal("100"),
        max_notional=Decimal("1000"),
        take_profit_pct=Decimal("0.5"),
        stop_loss_pct=Decimal("0.5"),
        time_stop_seconds=Decimal("0"),
    )
    strat = _ScriptedStrategy({0: low})
    result = await BacktestSimulator(strat, cfg).run_async([_trade(coin, 100, 1, "B", 0)])
    assert result.trades[0].notional == Decimal("100")  # confidence 0 -> min


# ------------------------------------------------------------------- metrics


def test_metrics_empty():
    m = compute_metrics([])
    assert m.trades == 0
    assert m.win_rate == 0.0


def test_metrics_basic_scorecard():
    trades = [_closed(10), _closed(-5), _closed(20), _closed(-5)]
    m = compute_metrics(trades)
    assert m.trades == 4
    assert m.wins == 2
    assert m.losses == 2
    assert m.win_rate == 0.5
    assert m.net_pnl == Decimal("20")
    assert m.gross_profit == Decimal("30")
    assert m.gross_loss == Decimal("10")
    assert m.profit_factor == 3.0
    assert m.expectancy == Decimal("5")


def test_metrics_max_drawdown():
    # Equity path: +10, +5(=15? no) ... use explicit losers to force a dip.
    trades = [_closed(10), _closed(-15), _closed(8)]
    m = compute_metrics(trades)
    # Curve: 10, -5, 3. Peak 10, trough -5 -> dd 15.
    assert m.max_drawdown == Decimal("15")


def test_metrics_profit_factor_infinite_when_no_losses():
    m = compute_metrics([_closed(10), _closed(5)])
    assert m.profit_factor == float("inf")


# --------------------------------------------------------------------- sweep


def test_default_grid_is_nonempty():
    grid = default_grid()
    assert grid
    assert "ret_z_entry" in grid


def test_sweep_runs_all_combos_and_ranks():
    # No signals expected from this tiny stream; sweep should still complete for
    # every grid point and produce a ranked list.
    events = [_trade("BTC", 100, 1, "B", i) for i in range(5)]
    grid = {
        "ret_z_entry": [Decimal("2.5"), Decimal("3.0")],
        "take_profit_pct": [Decimal("0.004"), Decimal("0.006")],
    }
    results = sweep(events, grid, min_trades=0)
    assert len(results) == 4  # 2 x 2 cartesian product
    assert all(r.metrics.trades == 0 for r in results)
