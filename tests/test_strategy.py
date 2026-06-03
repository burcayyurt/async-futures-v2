from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, MarketEvent, TradePayload
from src.strategy.momentum_oi import (
    MarketRegime,
    MomentumOIStrategy,
    SignalSide,
    StrategyParams,
    TradeSignal,
    _RegimeTracker,
)


@pytest.fixture
def strategy() -> MomentumOIStrategy:
    settings = HyperliquidSettings.from_env()
    params = StrategyParams(
        window_seconds=60,
        price_delta_pct=Decimal("0.8"),
        volume_spike_multiplier=Decimal("2.0"),
        oi_min_increase_pct=Decimal("0.05"),
        min_trades_in_window=3,
        ema_period=5,
    )
    strat = MomentumOIStrategy(settings, params=params)
    strat.seed_regime_prices(
        [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103"), Decimal("104")]
    )
    return strat


def test_strategy_params_from_settings() -> None:
    settings = HyperliquidSettings.from_env().model_copy(
        update={
            "strategy_price_delta_pct": Decimal("1.2"),
            "strategy_min_oi_increase_pct": Decimal("0.1"),
            "strategy_volume_spike_multiplier": Decimal("3.0"),
            "strategy_window_seconds": 90,
            "strategy_min_trades_in_window": 5,
            "strategy_ema_period": 15,
            "strategy_regime_coin": "ETH",
        }
    )
    params = StrategyParams.from_settings(settings)
    assert params.price_delta_pct == Decimal("1.2")
    assert params.oi_min_increase_pct == Decimal("0.1")
    assert params.volume_spike_multiplier == Decimal("3.0")
    assert params.window_seconds == 90
    assert params.min_trades_in_window == 5
    assert params.ema_period == 15
    assert params.regime_coin == "ETH"


def test_regime_hysteresis_prevents_ema_whipsaw() -> None:
    params = StrategyParams(ema_period=5, regime_buffer_pct=Decimal("0.15"))
    tracker = _RegimeTracker(params)
    for price in [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103"), Decimal("104")]:
        tracker.update(price)
    assert tracker.regime() == MarketRegime.BULLISH

    tracker.update(Decimal("103.95"))
    assert tracker.regime() == MarketRegime.BULLISH

    for price in [Decimal("99"), Decimal("98"), Decimal("97")]:
        tracker.update(price)
    assert tracker.regime() == MarketRegime.BEARISH


@pytest.mark.asyncio
async def test_breakout_allowed_without_volume_history(strategy: MomentumOIStrategy) -> None:
    coin = "SOL"
    await strategy.on_market_event(_ctx_event(coin, "100", "1000"))
    await strategy.on_market_event(_ctx_event(coin, "101", "1000"))
    await strategy.on_market_event(_trade_event(coin, "100", "5"))
    await strategy.on_market_event(_trade_event(coin, "100.5", "5"))
    await strategy.on_market_event(_trade_event(coin, "101", "5"))

    signal = await strategy.on_market_event(_ctx_event(coin, "101", "1001"))
    assert signal is not None
    assert signal.side == SignalSide.LONG


def _trade_event(coin: str, px: str, sz: str, ts: datetime | None = None) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.TRADE,
        coin=coin,
        ts=ts or datetime.now(timezone.utc),
        payload=TradePayload(
            px=Decimal(px),
            sz=Decimal(sz),
            side="B",
            hash="abc",
            tid=1,
        ),
    )


def _ctx_event(
    coin: str,
    mark_px: str,
    open_interest: str,
    ts: datetime | None = None,
) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.ASSET_CTX,
        coin=coin,
        ts=ts or datetime.now(timezone.utc),
        payload=AssetCtxPayload(
            mark_px=Decimal(mark_px),
            open_interest=Decimal(open_interest),
            funding=None,
            oracle_px=None,
            day_ntl_vlm=None,
        ),
    )


@pytest.mark.asyncio
async def test_regime_bullish_allows_long_blocks_short(strategy: MomentumOIStrategy) -> None:
    assert strategy.current_regime == MarketRegime.BULLISH

    long_signal = strategy.evaluate(
        "ETH",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("1.0"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1001"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("101"),
        side=SignalSide.LONG,
    )
    short_signal = strategy.evaluate(
        "ETH",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("1.0"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1001"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("99"),
        side=SignalSide.SHORT,
    )

    assert long_signal is not None
    assert long_signal.side == SignalSide.LONG
    assert short_signal is None


@pytest.mark.asyncio
async def test_regime_bearish_allows_short_blocks_long(strategy: MomentumOIStrategy) -> None:
    bearish = MomentumOIStrategy(
        HyperliquidSettings.from_env(),
        params=StrategyParams(ema_period=5),
    )
    bearish.seed_regime_prices(
        [Decimal("110"), Decimal("108"), Decimal("106"), Decimal("104"), Decimal("102")]
    )
    assert bearish.current_regime == MarketRegime.BEARISH

    short_signal = bearish.evaluate(
        "ETH",
        regime=MarketRegime.BEARISH,
        delta_pct=Decimal("1.0"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1001"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("99"),
        side=SignalSide.SHORT,
    )
    long_signal = bearish.evaluate(
        "ETH",
        regime=MarketRegime.BEARISH,
        delta_pct=Decimal("1.0"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1001"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("101"),
        side=SignalSide.LONG,
    )

    assert short_signal is not None
    assert short_signal.side == SignalSide.SHORT
    assert long_signal is None


def test_breakout_rejected_when_volume_low(strategy: MomentumOIStrategy) -> None:
    signal = strategy.evaluate(
        "SOL",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("1.5"),
        window_volume=Decimal("60"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1001"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("150"),
        side=SignalSide.LONG,
    )
    assert signal is None


def test_fakeout_rejected_when_oi_drops(strategy: MomentumOIStrategy) -> None:
    signal = strategy.evaluate(
        "SOL",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("1.5"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("999"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("150"),
        side=SignalSide.LONG,
    )
    assert signal is None


def test_signal_confirmed_when_oi_rises(strategy: MomentumOIStrategy) -> None:
    signal = strategy.evaluate(
        "SOL",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("1.5"),
        window_volume=Decimal("200"),
        avg_volume=Decimal("50"),
        open_interest=Decimal("1002"),
        prev_open_interest=Decimal("1000"),
        mark_px=Decimal("150"),
        side=SignalSide.LONG,
    )
    assert isinstance(signal, TradeSignal)
    assert signal.symbol == "SOL"
    assert signal.entry_mark_price == Decimal("150")
    assert signal.confidence > Decimal("0")


def test_evaluate_pure_function(strategy: MomentumOIStrategy) -> None:
    signal = strategy.evaluate(
        "BTC",
        regime=MarketRegime.BULLISH,
        delta_pct=Decimal("0.8"),
        window_volume=Decimal("1000"),
        avg_volume=Decimal("100"),
        open_interest=Decimal("5005"),
        prev_open_interest=Decimal("5000"),
        mark_px=Decimal("70000"),
        side=SignalSide.LONG,
        timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
    )
    assert signal is not None
    assert signal.reason == "momentum_oi_breakout_confirmed"
    assert signal.timestamp == datetime(2026, 5, 29, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_on_market_event_pending_breakout_confirmed_on_oi_update(
    strategy: MomentumOIStrategy,
) -> None:
    coin = "ETH"
    now = datetime.now(timezone.utc)
    state = strategy._coin_state(coin)
    state.prev_open_interest = Decimal("1000")
    state.open_interest = Decimal("1000")
    state.mark_px = Decimal("100")
    state.volume_history.append(Decimal("50"))

    from src.strategy.momentum_oi import _TradeSample

    state.trades.clear()
    for px in (Decimal("100"), Decimal("100.5"), Decimal("101.2")):
        state.trades.append(_TradeSample(ts=now, px=px, notional=px * Decimal("10")))

    state.mark_px = Decimal("101.2")
    candidate = strategy._detect_breakout(coin, state, now)
    assert candidate is not None
    state.pending_breakout = candidate
    state.open_interest = Decimal("1001.5")

    signal = await strategy.on_market_event(
        _ctx_event(coin, "101.2", "1001.5", ts=now)
    )
    assert signal is not None
    assert signal.side == SignalSide.LONG
