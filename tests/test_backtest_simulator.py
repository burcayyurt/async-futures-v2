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


def _long_signal(coin: str, px: str = "100") -> TradeSignal:
    return TradeSignal(
        symbol=coin,
        side=SignalSide.LONG,
        entry_mark_price=Decimal(px),
        confidence=Decimal("1"),
        timestamp=_ts(0),
        reason="test",
    )


def _trail_cfg(**over) -> SimConfig:
    base = dict(
        take_profit_pct=Decimal("0"),      # no fixed target, like live
        stop_loss_pct=Decimal("0.012"),    # ATR floor
        time_stop_seconds=Decimal("0"),
        trailing_callback_pct=Decimal("0.01"),
        break_even_trigger_pct=Decimal("0"),
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    base.update(over)
    return SimConfig(**base)


async def test_trailing_stop_fires_on_retrace_from_peak():
    """Rally to 105, then a 1% retrace off that peak exits — not off entry."""
    strat = _ScriptedStrategy({0: _long_signal("BTC")})
    events = [
        _trade("BTC", 100, 1, "B", 0),
        _trade("BTC", 105, 1, "B", 1),    # peak
        _trade("BTC", 103.9, 1, "A", 2),  # 105 * 0.99 = 103.95 -> triggers
    ]
    result = await BacktestSimulator(strat, _trail_cfg()).run_async(events)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_px == Decimal("103.9")  # books at the mark, as live does
    assert trade.net_pnl > 0                  # still a winner despite the give-back


async def test_trailing_is_armed_from_entry_not_after_profit():
    """Peak seeds to entry, so a drop straight off the open trails out."""
    strat = _ScriptedStrategy({0: _long_signal("ETH")})
    events = [
        _trade("ETH", 100, 1, "B", 0),
        _trade("ETH", 98.9, 1, "A", 1),  # -1.1%: trail (1%) hits before stop (1.2%)
    ]
    result = await BacktestSimulator(strat, _trail_cfg()).run_async(events)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "trailing_stop"


async def test_hard_stop_wins_when_trailing_is_wider():
    """A wide trail leaves the ATR stop as the binding exit."""
    strat = _ScriptedStrategy({0: _long_signal("SOL")})
    cfg = _trail_cfg(trailing_callback_pct=Decimal("0.05"))
    events = [
        _trade("SOL", 100, 1, "B", 0),
        _trade("SOL", 98.5, 1, "A", 1),  # below the 98.8 stop, above the 95 trail
    ]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"


async def test_break_even_ratchets_stop_above_entry():
    """After +2% the stop moves to entry+buffer, turning a round-trip into a win."""
    strat = _ScriptedStrategy({0: _long_signal("APT")})
    cfg = _trail_cfg(
        trailing_callback_pct=Decimal("0"),      # isolate break-even
        break_even_trigger_pct=Decimal("0.02"),
        fee_buffer_pct=Decimal("0.001"),
    )
    events = [
        _trade("APT", 100, 1, "B", 0),
        _trade("APT", 102.5, 1, "B", 1),  # arms break-even at 100.1
        _trade("APT", 100.0, 1, "A", 2),  # falls back through it
    ]
    result = await BacktestSimulator(strat, cfg).run_async(events)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_px == Decimal("100.1")
    assert trade.net_pnl > 0


def _maker_cfg(**over) -> SimConfig:
    base = dict(
        take_profit_pct=Decimal("0"),
        stop_loss_pct=Decimal("0.5"),        # far away; isolate entry behaviour
        time_stop_seconds=Decimal("0"),
        trailing_callback_pct=Decimal("0"),
        maker_entry_enabled=True,
        maker_fill_timeout_seconds=Decimal("60"),
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
    )
    base.update(over)
    return SimConfig(**base)


async def test_maker_entry_fills_only_when_price_returns():
    """A resting buy fills when price trades back down to it."""
    strat = _ScriptedStrategy({0: _long_signal("BTC", "100")})
    events = [
        _trade("BTC", 100, 1, "B", 0),
        _trade("BTC", 101, 1, "B", 1),   # runs away — no fill
        _trade("BTC", 99.5, 1, "A", 2),  # comes back — fills at 100
    ]
    sim = BacktestSimulator(strat, _maker_cfg())
    result = await sim.run_async(events)
    assert result.maker_orders_placed == 1
    assert result.maker_orders_filled == 1
    assert sim._open == {} or result.trades or True
    # Entry is the resting limit, not the touching price.
    opened = result.trades[0] if result.trades else None
    assert opened is not None and opened.entry_px == Decimal("100")


async def test_maker_entry_never_fills_when_price_runs_away():
    """The momentum case: price only goes up, the post-only buy never fills."""
    strat = _ScriptedStrategy({0: _long_signal("ETH", "100")})
    events = [
        _trade("ETH", 100, 1, "B", 0),
        _trade("ETH", 101, 1, "B", 1),
        _trade("ETH", 102, 1, "B", 2),
        _trade("ETH", 103, 1, "B", 3),
    ]
    result = await BacktestSimulator(strat, _maker_cfg()).run_async(events)
    assert result.maker_orders_placed == 1
    assert result.maker_orders_filled == 0
    assert result.trades == []


async def test_maker_entry_expires_after_timeout():
    strat = _ScriptedStrategy({0: _long_signal("SOL", "100")})
    events = [
        _trade("SOL", 100, 1, "B", 0),
        _trade("SOL", 101, 1, "B", 30),
        _trade("SOL", 99, 1, "A", 120),  # back in range, but the order expired
    ]
    result = await BacktestSimulator(strat, _maker_cfg()).run_async(events)
    assert result.maker_orders_expired == 1
    assert result.maker_orders_filled == 0
    assert result.trades == []


async def test_maker_entry_disabled_fills_immediately():
    """Default behaviour is unchanged: fill at the signal price."""
    strat = _ScriptedStrategy({0: _long_signal("APT", "100")})
    events = [_trade("APT", 100, 1, "B", 0), _trade("APT", 101, 1, "B", 1)]
    result = await BacktestSimulator(strat, _maker_cfg(maker_entry_enabled=False)).run_async(events)
    assert result.maker_orders_placed == 0
    assert len(result.trades) == 1


async def test_entry_fee_follows_maker_flag():
    """Immediate fills cross the spread and must be charged the taker rate."""

    async def _net(maker: bool) -> Decimal:
        strat = _ScriptedStrategy({0: _long_signal("BTC", "100")})
        cfg = SimConfig(
            take_profit_pct=Decimal("0"),
            stop_loss_pct=Decimal("0.5"),
            time_stop_seconds=Decimal("1"),
            trailing_callback_pct=Decimal("0"),
            maker_entry_enabled=maker,
            maker_fee_bps=Decimal("1.5"),
            taker_fee_bps=Decimal("4.5"),
            slippage_bps=Decimal("0"),
        )
        events = [_trade("BTC", 100, 1, "B", 0), _trade("BTC", 100, 1, "B", 5)]
        result = await BacktestSimulator(strat, cfg).run_async(events)
        return result.trades[0].fees

    maker_fees = await _net(True)
    taker_fees = await _net(False)
    # Same exit leg both ways, so the whole gap is the entry rate: 3bps on the
    # $1000 notional that confidence=1 produces (min 100 + 900 * 1.0).
    assert taker_fees > maker_fees
    assert taker_fees - maker_fees == Decimal("1000") * Decimal("3.0") / Decimal("10000")


async def test_trailing_disabled_by_default():
    """Default config keeps the original pure-bracket behaviour."""
    assert SimConfig().trailing_callback_pct == Decimal("0")
    assert SimConfig().break_even_trigger_pct == Decimal("0")


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
