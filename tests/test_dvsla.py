"""Tests for the DVSLA liquidation-cascade mean-reversion engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.dvsla import (
    CascadeDirection,
    CascadeSignal,
    DvslaParams,
    DvslaStrategy,
)
from src.strategy.signals import SignalSide

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(step: int) -> datetime:
    return BASE_TS + timedelta(milliseconds=step)


def _trade(coin: str, px: float, sz: float, side: str, step: int) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.TRADE,
        coin=coin,
        ts=_ts(step),
        payload=TradePayload(
            px=Decimal(str(px)),
            sz=Decimal(str(sz)),
            side=side,
            hash="0xabc",
            tid=None,
        ),
    )


def _ctx(coin: str, mark: float, oi: float, oracle: float, step: int) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.ASSET_CTX,
        coin=coin,
        ts=_ts(step),
        payload=AssetCtxPayload(
            mark_px=Decimal(str(mark)),
            open_interest=Decimal(str(oi)),
            funding=None,
            oracle_px=Decimal(str(oracle)),
            day_ntl_vlm=None,
        ),
    )


def _fast_params(**overrides) -> DvslaParams:
    base = dict(
        volume_bar_threshold=Decimal("3"),
        ret_z_window=10,
        ret_z_entry=Decimal("2.0"),
        flow_window=3,
        flow_imbalance_min=Decimal("0.5"),
        oi_z_window=10,
        oi_z_drop=Decimal("-1.5"),
        hurst_window=32,
        hurst_max=Decimal("0.45"),
        warmup_bars=20,
        cooldown_bars=5,
    )
    base.update(overrides)
    return DvslaParams(**base)


async def _feed(strategy: DvslaStrategy, events: list[MarketEvent]) -> list:
    signals = []
    for event in events:
        sig = await strategy.on_market_event(event)
        if sig is not None:
            signals.append(sig)
    return signals


def _warmup_events(coin: str, *, trending: bool = False) -> tuple[list[MarketEvent], float, int]:
    """Build oscillating (mean-reverting) or trending warmup bars.

    Each bar is exactly three size-1 trades so it closes cleanly with no
    residual. Returns the event list, the final close price, and the next time
    step to continue from.
    """

    events: list[MarketEvent] = []
    close = 100.0
    step = 0
    oi = 100_000.0
    sides = ("B", "A", "B")

    for i in range(22):
        open_px = close
        if trending:
            # Persistent (positively autocorrelated) returns: they drift
            # monotonically rather than oscillate, so the *return* series is
            # trending (high Hurst) and the regime gate stands down.
            ret = 0.001 + 0.0004 * i
        else:
            # Anti-persistent oscillation -> strongly mean-reverting (low Hurst).
            ret = 0.002 if i % 2 == 0 else -0.002
        close = open_px * (1.0 + ret)
        mid = (open_px + close) / 2.0

        events.append(_trade(coin, open_px, 1, sides[0], step))
        step += 1
        events.append(_trade(coin, mid, 1, sides[1], step))
        step += 1
        events.append(_trade(coin, close, 1, sides[2], step))
        step += 1

        # Alternating OI deltas build a non-degenerate change distribution.
        oi += 10.0 if i % 2 == 0 else -10.0
        events.append(_ctx(coin, close, oi, close, step))
        step += 1

    return events, close, step


def _down_cascade_events(coin: str, last_close: float, step: int) -> list[MarketEvent]:
    """A sharp sell-driven drop with collapsing OI (a liquidation cascade)."""

    events: list[MarketEvent] = []
    # OI collapses hard right before the print bar closes.
    events.append(_ctx(coin, last_close, 99_000.0, last_close, step))
    step += 1

    open_px = last_close
    floor_px = last_close * 0.97
    mid_px = (open_px + floor_px) / 2.0
    for px in (open_px, mid_px, floor_px):
        events.append(_trade(coin, px, 1, "A", step))
        step += 1
    return events


def _osc_warmup(
    coin: str, amp: float, *, step: int = 0, close: float = 100.0, n: int = 22
) -> tuple[list[MarketEvent], float, int]:
    """Oscillating (mean-reverting) warmup with a tunable return amplitude.

    A small ``amp`` collapses the return variance, so a later move can produce a
    huge z-score even though the move itself is tiny — exactly the variance-
    collapse footprint the robustness guards must neutralize.
    """

    events: list[MarketEvent] = []
    oi = 100_000.0
    sides = ("B", "A", "B")
    for i in range(n):
        open_px = close
        ret = amp if i % 2 == 0 else -amp
        close = open_px * (1.0 + ret)
        mid = (open_px + close) / 2.0
        events.append(_trade(coin, open_px, 1, sides[0], step))
        step += 1
        events.append(_trade(coin, mid, 1, sides[1], step))
        step += 1
        events.append(_trade(coin, close, 1, sides[2], step))
        step += 1
        oi += 10.0 if i % 2 == 0 else -10.0
        events.append(_ctx(coin, close, oi, close, step))
        step += 1
    return events, close, step


def _sell_bar(coin: str, last_close: float, drop_frac: float, step: int) -> list[MarketEvent]:
    """A single sell-driven down bar of size ``drop_frac`` (fraction)."""

    events: list[MarketEvent] = []
    events.append(_ctx(coin, last_close, 99_000.0, last_close, step))
    step += 1
    open_px = last_close
    floor_px = last_close * (1.0 - drop_frac)
    mid_px = (open_px + floor_px) / 2.0
    for px in (open_px, mid_px, floor_px):
        events.append(_trade(coin, px, 1, "A", step))
        step += 1
    return events


# --------------------------------------------------------------------- units


def test_threshold_for_uses_override():
    params = DvslaParams(
        volume_bar_threshold=Decimal("5"),
        symbol_thresholds={"BTC": Decimal("50")},
    )
    assert params.threshold_for("btc") == Decimal("50")
    assert params.threshold_for("eth") == Decimal("5")


@pytest.mark.parametrize(
    "ret_z, expected",
    [
        (-3.5, CascadeDirection.DOWN),
        (3.5, CascadeDirection.UP),
        (1.0, None),
        (-1.0, None),
    ],
)
def test_classify_cascade(ret_z, expected):
    strat = DvslaStrategy(_fast_params(ret_z_entry=Decimal("2.0")))
    assert strat._classify_cascade(ret_z) == expected


def test_confidence_within_bounds():
    strat = DvslaStrategy(_fast_params())
    cascade = CascadeSignal(
        direction=CascadeDirection.DOWN,
        ret_z=-8.0,
        imbalance=-1.0,
        oi_z=-6.0,
        hurst=0.2,
        divergence_pct=-0.4,
    )
    conf = strat._confidence(cascade)
    assert Decimal("0") <= conf <= Decimal("1")
    assert conf > Decimal("0.5")


def test_divergence_pct_handles_missing_oracle():
    strat = DvslaStrategy(_fast_params())
    state = strat._state("BTC")
    assert strat._divergence_pct(state) == 0.0


# --------------------------------------------------------------- integration


async def test_down_cascade_emits_long():
    coin = "ETH"
    strat = DvslaStrategy(_fast_params())
    warmup, last_close, step = _warmup_events(coin)
    cascade = _down_cascade_events(coin, last_close, step)

    warmup_signals = await _feed(strat, warmup)
    assert warmup_signals == []  # nothing fires during the calm regime

    signals = await _feed(strat, cascade)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == coin
    assert signal.side == SignalSide.LONG  # we fade the down-move
    assert "dvsla_fade_down" in signal.reason
    assert Decimal("0") < signal.confidence <= Decimal("1")


async def test_invert_trades_with_the_cascade():
    coin = "ETH"
    strat = DvslaStrategy(_fast_params(invert=True))
    warmup, last_close, step = _warmup_events(coin)
    cascade = _down_cascade_events(coin, last_close, step)

    await _feed(strat, warmup)
    signals = await _feed(strat, cascade)
    assert len(signals) == 1
    signal = signals[0]
    # Inverted: a down-cascade is now SOLD (momentum continuation), not bought.
    assert signal.side == SignalSide.SHORT
    assert "dvsla_momentum_down" in signal.reason


async def test_signal_callback_invoked():
    coin = "SOL"
    captured = []
    strat = DvslaStrategy(_fast_params(), on_signal=captured.append)
    warmup, last_close, step = _warmup_events(coin)
    cascade = _down_cascade_events(coin, last_close, step)

    await _feed(strat, warmup)
    await _feed(strat, cascade)
    assert len(captured) == 1
    assert captured[0].side == SignalSide.LONG


async def test_trending_regime_blocks_signal():
    coin = "AVAX"
    strat = DvslaStrategy(_fast_params())
    warmup, last_close, step = _warmup_events(coin, trending=True)
    cascade = _down_cascade_events(coin, last_close, step)

    await _feed(strat, warmup)
    signals = await _feed(strat, cascade)
    assert signals == []  # Hurst gate stands down in a trending regime


async def test_cooldown_blocks_second_cascade():
    coin = "INJ"
    strat = DvslaStrategy(_fast_params(cooldown_bars=50))
    warmup, last_close, step = _warmup_events(coin)
    first = _down_cascade_events(coin, last_close, step)

    await _feed(strat, warmup)
    first_signals = await _feed(strat, first)
    assert len(first_signals) == 1

    # A second identical cascade immediately after must be suppressed.
    floor_px = last_close * 0.97
    second = _down_cascade_events(coin, floor_px, step + 100)
    second_signals = await _feed(strat, second)
    assert second_signals == []


async def test_min_abs_return_filters_micro_move():
    # A sub-threshold move (-0.1% < 0.15%) against a collapsed-variance window
    # produces an extreme z-score but is NOT a real cascade -> must be skipped.
    coin = "ETH"
    strat = DvslaStrategy(_fast_params(ret_min_abs_pct=Decimal("0.0015")))
    warmup, last_close, step = _osc_warmup(coin, 0.0002)
    micro = _sell_bar(coin, last_close, 0.001, step)

    await _feed(strat, warmup)
    assert await _feed(strat, micro) == []


async def test_min_abs_return_allows_real_move():
    # The same setup but with a real -0.3% move clears the floor and fires.
    coin = "ETH"
    strat = DvslaStrategy(_fast_params(ret_min_abs_pct=Decimal("0.0015")))
    warmup, last_close, step = _osc_warmup(coin, 0.0002)
    real = _sell_bar(coin, last_close, 0.003, step)

    await _feed(strat, warmup)
    signals = await _feed(strat, real)
    assert len(signals) == 1
    assert signals[0].side == SignalSide.LONG


async def test_ret_z_is_clamped_to_the_configured_cap():
    # Collapsed variance + a >0.15% move yields a raw z far beyond the cap; the
    # signal must carry the clamped value (±25), never an astronomical z-score.
    coin = "ETH"
    strat = DvslaStrategy(
        _fast_params(ret_z_clamp=Decimal("25"), ret_min_abs_pct=Decimal("0.0015"))
    )
    warmup, last_close, step = _osc_warmup(coin, 0.0002)
    shock = _sell_bar(coin, last_close, 0.006, step)

    await _feed(strat, warmup)
    signals = await _feed(strat, shock)
    assert len(signals) == 1
    assert "ret_z=-25.00" in signals[0].reason


async def test_oi_collapse_raises_confidence_but_is_not_a_gate():
    # OI is a confidence weight, not a hard veto: a cascade with steady OI still
    # fires, but with lower confidence than the same cascade with an OI collapse.
    def run(oi_collapse: bool):
        coin = "SUI"
        strat = DvslaStrategy(_fast_params())
        warmup, last_close, step = _warmup_events(coin)
        events: list[MarketEvent] = []
        oi = 99_000.0 if oi_collapse else 100_010.0
        events.append(_ctx(coin, last_close, oi, last_close, step))
        step += 1
        open_px = last_close
        floor_px = last_close * 0.97
        mid_px = (open_px + floor_px) / 2.0
        for px in (open_px, mid_px, floor_px):
            events.append(_trade(coin, px, 1, "A", step))
            step += 1
        return warmup, events, strat

    # Both cascades emit a signal; the OI-collapse one is at least as confident.
    async def _collect(warmup, events, strat):
        return await _feed(strat, warmup) + await _feed(strat, events)

    w1, e1, s1 = run(True)
    sig_collapse = await _collect(w1, e1, s1)
    w2, e2, s2 = run(False)
    sig_steady = await _collect(w2, e2, s2)

    assert len(sig_collapse) == 1
    assert len(sig_steady) == 1
    assert sig_collapse[0].confidence >= sig_steady[0].confidence


# --------------------------------------------------------------------------
# PR6 — engine wiring (params from settings, decision summary, selector)
# --------------------------------------------------------------------------


def test_dvsla_params_from_settings_maps_env_fields():
    from src.core.config import HyperliquidSettings

    settings = HyperliquidSettings.from_env().model_copy(
        update={
            "dvsla_volume_bar_threshold": Decimal("123"),
            "dvsla_ret_z_entry": Decimal("2.5"),
            "dvsla_ret_min_abs_pct": Decimal("0.002"),
            "dvsla_ret_z_clamp": Decimal("30"),
            "dvsla_flow_imbalance_min": Decimal("0.6"),
            "dvsla_oi_z_drop": Decimal("-1.8"),
            "dvsla_hurst_max": Decimal("0.4"),
            "dvsla_warmup_bars": 7,
            "dvsla_cooldown_bars": 3,
            "dvsla_invert": True,
        }
    )
    params = DvslaParams.from_settings(settings)
    assert params.invert is True
    assert params.volume_bar_threshold == Decimal("123")
    assert params.ret_z_entry == Decimal("2.5")
    assert params.ret_min_abs_pct == Decimal("0.002")
    assert params.ret_z_clamp == Decimal("30")
    assert params.flow_imbalance_min == Decimal("0.6")
    assert params.oi_z_drop == Decimal("-1.8")
    assert params.hurst_max == Decimal("0.4")
    assert params.warmup_bars == 7
    assert params.cooldown_bars == 3


def test_default_strategy_engine_is_dvsla():
    from src.core.config import HyperliquidSettings

    assert HyperliquidSettings.from_env().strategy_engine == "dvsla"


def test_symbol_thresholds_calibrated_without_dead_feeds():
    from src.core.config import HyperliquidSettings

    settings = HyperliquidSettings.from_env()
    thresholds = settings.dvsla_symbol_thresholds

    # Dead feeds (renamed to POL/RENDER on Hyperliquid) must be gone.
    assert "MATIC" not in thresholds
    assert "RNDR" not in thresholds
    assert "MATIC" not in settings.symbols
    assert "RNDR" not in settings.symbols

    # Every watched symbol has a calibrated threshold (no silent tiny fallback
    # that would spawn thousands of micro-bars/sec).
    for symbol in settings.symbols:
        assert symbol in thresholds, f"{symbol} missing a calibrated threshold"
    # Sanity: the worst offender from the incident is now sized sanely.
    assert thresholds["WLD"] >= Decimal("1000")


async def test_decision_summary_reports_idle_warmup_and_active():
    coin = "BTC"
    strat = DvslaStrategy(_fast_params(warmup_bars=20))

    # No data yet for the watched symbol -> idle.
    summary = strat.decision_summary(("BTC", "ETH"))
    assert "engine=dvsla" in summary
    assert "BTC:idle" in summary

    warmup, _last_close, _step = _warmup_events(coin)
    await _feed(strat, warmup)
    summary = strat.decision_summary(("BTC",))
    # After warmup bars are fed the symbol is past warmup and reports state.
    assert "BTC:bars=" in summary
    assert "oi_z=" in summary

