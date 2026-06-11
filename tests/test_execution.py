from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.core.config import HyperliquidSettings
from src.core.open_position_store import OpenPositionStore
from src.core.trade_journal import TradeJournal
from src.execution.order_router import OrderRouter
from src.execution.position_manager import ManagedPosition, PositionManager
from src.risk.kill_switch import KillSwitch
from src.risk.margin_manager import MarginManager, MarginSnapshot
from src.strategy.momentum_oi import SignalSide, TradeSignal


@pytest.fixture
def settings() -> HyperliquidSettings:
    return HyperliquidSettings.from_env().model_copy(
        update={
            "bot_dry_run": True,
            "trade_risk_pct": Decimal("1.0"),
            "trailing_callback_pct": Decimal("0.006"),
            "break_even_trigger_pct": Decimal("0.5"),
            # Pin PR5 execution upgrades OFF so the base fixture is independent
            # of the live .env (tests opt in per-case via model_copy).
            "maker_entry_enabled": False,
            "confidence_sizing_enabled": False,
            "atr_stop_enabled": False,
            "reversion_tp_enabled": False,
            "reentry_cooldown_seconds": 0,
        }
    )


@pytest.fixture
def kill_switch(settings: HyperliquidSettings) -> KillSwitch:
    return KillSwitch(settings)


@pytest.fixture
def position_manager(settings: HyperliquidSettings, kill_switch: KillSwitch) -> PositionManager:
    return PositionManager(settings, kill_switch)


@pytest.fixture
def router(
    settings: HyperliquidSettings,
    kill_switch: KillSwitch,
    position_manager: PositionManager,
) -> OrderRouter:
    rest = MagicMock()
    rest.place_order = AsyncMock(return_value={"status": "dry_run"})
    margin_manager = MagicMock(spec=MarginManager)
    margin_manager.fetch_margin_snapshot = AsyncMock(
        return_value=MarginSnapshot(
            equity=Decimal("1000"),
            used_margin=Decimal("0"),
            available_margin=Decimal("1000"),
            leverage=10,
        )
    )
    margin_manager.calculate_position_size = MagicMock(return_value=0.004)
    return OrderRouter(settings, rest, kill_switch, margin_manager, position_manager)


def _signal(side: SignalSide = SignalSide.LONG, symbol: str = "BTC") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        side=side,
        entry_mark_price=Decimal("50000"),
        confidence=Decimal("0.8"),
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_route_entry_rejects_when_tripped(router: OrderRouter, kill_switch: KillSwitch) -> None:
    await kill_switch.trip("test")
    result = await router.route_entry(_signal())
    assert result is None
    router.rest.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_route_entry_places_market_order_and_registers(router: OrderRouter) -> None:
    result = await router.route_entry(_signal())
    assert result is not None
    router.rest.place_order.assert_awaited_once()
    order = router.rest.place_order.await_args.args[0]
    assert order.coin == "BTC"
    assert order.is_buy is True
    assert order.tif == "Ioc"
    assert "BTC" in router.position_manager.positions


@pytest.mark.asyncio
async def test_position_break_even_arms_stop_long(
    position_manager: PositionManager,
) -> None:
    position = ManagedPosition(
        coin="ETH",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("98"),
        peak_px=Decimal("100"),
    )
    await position_manager.register_position(position)

    new_stop = await position_manager.maybe_move_to_break_even(position, Decimal("100.6"))
    assert new_stop is not None
    assert position.break_even_armed is True
    assert new_stop > Decimal("100")


@pytest.mark.asyncio
async def test_position_trailing_exit_long(position_manager: PositionManager) -> None:
    exits: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        exits.append(reason)
        await position_manager.remove_position(position.coin)

    position_manager.bind_exit_handler(handler)
    position = ManagedPosition(
        coin="ETH",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("98"),
        peak_px=Decimal("101"),
    )
    await position_manager.register_position(position)

    await position_manager.on_price_update("ETH", Decimal("100.3"))
    assert exits == ["trailing_stop"]
    assert "ETH" not in position_manager.positions


@pytest.mark.asyncio
async def test_panic_close_all_on_kill_switch(
    position_manager: PositionManager,
    kill_switch: KillSwitch,
) -> None:
    closed: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        closed.append(position.coin)
        await position_manager.remove_position(position.coin)

    position_manager.bind_exit_handler(handler)
    await position_manager.register_position(
        ManagedPosition(
            coin="BTC",
            side=SignalSide.LONG,
            entry_px=Decimal("50000"),
            size=Decimal("0.01"),
            stop_px=Decimal("49000"),
            peak_px=Decimal("50000"),
        )
    )
    await kill_switch.trip("panic test")
    await position_manager.on_price_update("BTC", Decimal("49900"))

    assert closed == ["BTC"]


@pytest.mark.asyncio
async def test_short_trailing_exit(position_manager: PositionManager) -> None:
    exits: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        exits.append(reason)
        await position_manager.remove_position(position.coin)

    position_manager.bind_exit_handler(handler)
    position = ManagedPosition(
        coin="SOL",
        side=SignalSide.SHORT,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("102"),
        peak_px=Decimal("99"),
    )
    await position_manager.register_position(position)

    await position_manager.on_price_update("SOL", Decimal("99.7"))
    assert exits == ["trailing_stop"]


@pytest.mark.asyncio
async def test_trailing_exit_records_closed_trade(
    settings: HyperliquidSettings,
    kill_switch: KillSwitch,
) -> None:
    journal = Mock(spec=TradeJournal)
    journal.record_closed_trade = AsyncMock()
    position_manager = PositionManager(settings, kill_switch, trade_journal=journal)
    exits: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        exits.append(reason)
        await position_manager.remove_position(position.coin)

    position_manager.bind_exit_handler(handler)
    position = ManagedPosition(
        coin="ETH",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("98"),
        peak_px=Decimal("101"),
    )
    await position_manager.register_position(position)

    await position_manager.on_price_update("ETH", Decimal("100.3"))

    assert exits == ["trailing_stop"]
    journal.record_closed_trade.assert_called_once()
    call_args = journal.record_closed_trade.call_args
    assert call_args.args[0].coin == "ETH"
    assert call_args.args[4] == "trailing_stop"


@pytest.mark.asyncio
async def test_register_and_remove_persist_open_position_store(
    settings: HyperliquidSettings,
    kill_switch: KillSwitch,
    tmp_path,
) -> None:
    store_path = tmp_path / "open_positions.json"
    store = OpenPositionStore(
        settings.model_copy(update={"open_positions_path": str(store_path)}),
        path=store_path,
    )
    position_manager = PositionManager(settings, kill_switch, open_position_store=store)
    position = ManagedPosition(
        coin="LINK",
        side=SignalSide.LONG,
        entry_px=Decimal("9"),
        size=Decimal("1"),
        stop_px=Decimal("8.5"),
        peak_px=Decimal("9"),
    )
    await position_manager.register_position(position)
    loaded = await store.load()
    assert "LINK" in loaded

    await position_manager.remove_position("LINK")
    loaded = await store.load()
    assert "LINK" not in loaded


@pytest.mark.asyncio
async def test_recover_positions_filters_symbols_outside_watchlist(
    settings: HyperliquidSettings,
    kill_switch: KillSwitch,
    tmp_path,
) -> None:
    store_path = tmp_path / "open_positions.json"
    store = OpenPositionStore(
        settings.model_copy(update={"open_positions_path": str(store_path)}),
        path=store_path,
    )
    await store.upsert(
        ManagedPosition(
            coin="ZZZ",
            side=SignalSide.LONG,
            entry_px=Decimal("1"),
            size=Decimal("1"),
            stop_px=Decimal("0.9"),
            peak_px=Decimal("1"),
        )
    )
    position_manager = PositionManager(settings, kill_switch, open_position_store=store)
    recovered = await position_manager.recover_positions()
    assert recovered == []
    assert "ZZZ" not in position_manager.positions


# --------------------------------------------------------------------------
# PR5 execution upgrades
# --------------------------------------------------------------------------


def _build_router(settings: HyperliquidSettings) -> OrderRouter:
    kill_switch = KillSwitch(settings)
    position_manager = PositionManager(settings, kill_switch)
    rest = MagicMock()
    rest.place_order = AsyncMock(return_value={"status": "dry_run"})
    margin_manager = MagicMock(spec=MarginManager)
    margin_manager.fetch_margin_snapshot = AsyncMock(
        return_value=MarginSnapshot(
            equity=Decimal("1000"),
            used_margin=Decimal("0"),
            available_margin=Decimal("1000"),
            leverage=10,
        )
    )
    margin_manager.calculate_position_size = MagicMock(return_value=1.0)
    return OrderRouter(settings, rest, kill_switch, margin_manager, position_manager)


@pytest.mark.asyncio
async def test_maker_entry_uses_alo_at_signal_price(settings: HyperliquidSettings) -> None:
    router = _build_router(settings.model_copy(update={"maker_entry_enabled": True}))
    await router.route_entry(_signal(SignalSide.LONG))
    order = router.rest.place_order.await_args.args[0]
    assert order.tif == "Alo"
    assert order.limit_px == "50000"  # passive, no spread cross


@pytest.mark.asyncio
async def test_confidence_sizing_scales_size(settings: HyperliquidSettings) -> None:
    router = _build_router(
        settings.model_copy(
            update={
                "confidence_sizing_enabled": True,
                "confidence_size_floor": Decimal("0.5"),
            }
        )
    )
    # confidence 0.8, floor 0.5 -> factor = 0.5 + 0.5*0.8 = 0.9 ; base size 1.0
    await router.route_entry(_signal(SignalSide.LONG))
    order = router.rest.place_order.await_args.args[0]
    assert abs(float(order.sz) - 0.9) < 1e-9


@pytest.mark.asyncio
async def test_reentry_cooldown_blocks_immediate_reentry(
    settings: HyperliquidSettings,
) -> None:
    router = _build_router(
        settings.model_copy(update={"reentry_cooldown_seconds": 60})
    )
    # Simulate a recent exit on BTC.
    from datetime import datetime, timezone

    router._last_exit_at["BTC"] = datetime.now(timezone.utc)
    result = await router.route_entry(_signal(SignalSide.LONG, symbol="BTC"))
    assert result is None
    router.rest.place_order.assert_not_called()


def test_atr_stop_falls_back_when_cold(settings: HyperliquidSettings) -> None:
    pm = PositionManager(settings.model_copy(update={"atr_stop_enabled": True}), KillSwitch(settings))
    # No mark history yet -> fixed-pct fallback.
    stop = pm.compute_initial_stop("BTC", Decimal("100"), SignalSide.LONG)
    assert stop == Decimal("100") * (Decimal("1") - Decimal("0.02"))


@pytest.mark.asyncio
async def test_atr_stop_uses_volatility_band(settings: HyperliquidSettings) -> None:
    cfg = settings.model_copy(
        update={"atr_stop_enabled": True, "atr_window": 5, "atr_stop_mult": Decimal("3")}
    )
    pm = PositionManager(cfg, KillSwitch(cfg))
    # Feed a few mark prints so the ATR proxy warms up.
    for px in ("100", "101", "100", "102", "101"):
        await pm.on_price_update("BTC", Decimal(px))
    stop = pm.compute_initial_stop("BTC", Decimal("100"), SignalSide.LONG)
    # ATR-based stop must sit below entry but be a different distance than the
    # fixed 2% fallback.
    assert stop < Decimal("100")
    assert stop != Decimal("100") * (Decimal("1") - Decimal("0.02"))


@pytest.mark.asyncio
async def test_atr_stop_floored_at_min_pct(settings: HyperliquidSettings) -> None:
    # Per-tick mark returns are microscopic, so the raw ATR proxy would place the
    # stop right on top of the entry. The floor keeps it at least min_pct away.
    cfg = settings.model_copy(
        update={
            "atr_stop_enabled": True,
            "atr_window": 5,
            "atr_stop_mult": Decimal("2.5"),
            "atr_stop_min_pct": Decimal("0.015"),
        }
    )
    pm = PositionManager(cfg, KillSwitch(cfg))
    # Tiny tick moves (~0.01%) -> atr_pct * mult stays far below the floor.
    for px in ("100", "100.01", "100.0", "100.01", "100.0"):
        await pm.on_price_update("BTC", Decimal(px))
    long_stop = pm.compute_initial_stop("BTC", Decimal("100"), SignalSide.LONG)
    short_stop = pm.compute_initial_stop("BTC", Decimal("100"), SignalSide.SHORT)
    assert long_stop == Decimal("100") * (Decimal("1") - Decimal("0.015"))
    assert short_stop == Decimal("100") * (Decimal("1") + Decimal("0.015"))


@pytest.mark.asyncio
async def test_reversion_take_profit_exits_long_at_mean(
    settings: HyperliquidSettings,
) -> None:
    cfg = settings.model_copy(
        update={"reversion_tp_enabled": True, "reversion_window": 10}
    )
    pm = PositionManager(cfg, KillSwitch(cfg))
    exits: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        exits.append(reason)
        await pm.remove_position(position.coin)

    pm.bind_exit_handler(handler)
    # Build a reversion target (rolling mean) around ~100.
    for px in ("100", "100", "100", "98"):
        await pm.on_price_update("ETH", Decimal(px))

    position = ManagedPosition(
        coin="ETH",
        side=SignalSide.LONG,
        entry_px=Decimal("98"),  # faded a dip below the mean
        size=Decimal("1"),
        stop_px=Decimal("96"),
        peak_px=Decimal("98"),
    )
    await pm.register_position(position)
    # Price recovers back up to/above the mean -> reversion take-profit.
    await pm.on_price_update("ETH", Decimal("100.5"))
    assert exits == ["reversion_tp"]


