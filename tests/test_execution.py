from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.core.config import HyperliquidSettings
from src.core.open_position_store import OpenPositionStore
from src.core.trade_journal import TradeJournal
from src.exchange.hyperliquid_rest import ExchangePosition
from src.execution.order_router import OrderRouter
from src.execution.position_manager import (
    ManagedPosition,
    PositionManager,
    PositionReconciliationError,
)
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
            "dvsla_min_confidence": Decimal("0"),
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


def _signal(
    side: SignalSide = SignalSide.LONG,
    symbol: str = "BTC",
    confidence: Decimal = Decimal("0.8"),
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        side=side,
        entry_mark_price=Decimal("50000"),
        confidence=confidence,
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
async def test_panic_close_continues_after_a_failing_symbol(
    position_manager: PositionManager,
    kill_switch: KillSwitch,
) -> None:
    """One symbol failing to close must not strand the others.

    A panic close runs precisely when things are going wrong, so the symbol that
    errors is often the one that is moving hardest.
    """
    attempted: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        attempted.append(position.coin)
        if position.coin == "ETH":
            raise ConnectionError("exchange rejected the close")
        await position_manager.remove_position(position.coin)

    position_manager.bind_exit_handler(handler)
    for coin in ("BTC", "ETH", "SOL"):
        await position_manager.register_position(
            ManagedPosition(
                coin=coin,
                side=SignalSide.LONG,
                entry_px=Decimal("100"),
                size=Decimal("1"),
                stop_px=Decimal("98"),
                peak_px=Decimal("100"),
            )
        )

    await kill_switch.trip("panic test")
    await position_manager.on_price_update("BTC", Decimal("99"))

    assert sorted(attempted) == ["BTC", "ETH", "SOL"]  # every symbol attempted
    assert "BTC" not in position_manager.positions
    assert "SOL" not in position_manager.positions
    assert "ETH" in position_manager.positions  # the failure stays open, and visible


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
# Horizon (time) exit
# --------------------------------------------------------------------------


async def _pm_with_max_hold(
    settings: HyperliquidSettings, seconds: str, *, trailing: str = "0"
) -> tuple[PositionManager, list[str]]:
    tuned = settings.model_copy(
        update={
            "max_hold_seconds": Decimal(seconds),
            "trailing_callback_pct": Decimal(trailing),
            "break_even_trigger_pct": Decimal("0"),
        }
    )
    pm = PositionManager(tuned, KillSwitch(tuned))
    reasons: list[str] = []

    async def handler(position: ManagedPosition, reason: str) -> None:
        reasons.append(reason)
        await pm.remove_position(position.coin)

    pm.bind_exit_handler(handler)
    return pm, reasons


def _aged_position(seconds_ago: int) -> ManagedPosition:
    return ManagedPosition(
        coin="BTC",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("90"),
        peak_px=Decimal("100"),
        opened_at=datetime.now(timezone.utc) - timedelta(seconds=seconds_ago),
    )


@pytest.mark.asyncio
async def test_zero_callback_disables_the_trail(settings: HyperliquidSettings) -> None:
    """0 must mean "no trail", not "trail sitting exactly on the peak".

    With the trigger equal to the peak, every tick that is not a new high closes
    the position — which would silently wreck the horizon-exit configuration.
    """
    pm, reasons = await _pm_with_max_hold(settings, "0", trailing="0")
    await pm.register_position(_aged_position(5))
    await pm.on_price_update("BTC", Decimal("99.99"))  # below peak, not a new high
    assert reasons == []
    assert "BTC" in pm.positions


@pytest.mark.asyncio
async def test_max_hold_closes_a_stale_position(settings: HyperliquidSettings) -> None:
    pm, reasons = await _pm_with_max_hold(settings, "60")
    await pm.register_position(_aged_position(120))
    await pm.on_price_update("BTC", Decimal("100.05"))
    assert reasons == ["time_stop"]


@pytest.mark.asyncio
async def test_max_hold_leaves_a_fresh_position_open(settings: HyperliquidSettings) -> None:
    pm, reasons = await _pm_with_max_hold(settings, "60")
    await pm.register_position(_aged_position(5))
    await pm.on_price_update("BTC", Decimal("100.05"))
    assert reasons == []
    assert "BTC" in pm.positions


@pytest.mark.asyncio
async def test_max_hold_disabled_by_default_keeps_position(
    settings: HyperliquidSettings,
) -> None:
    """Zero must preserve today's behaviour, or enabling the code changes the run."""
    pm, reasons = await _pm_with_max_hold(settings, "0")
    await pm.register_position(_aged_position(100_000))
    await pm.on_price_update("BTC", Decimal("100.05"))
    assert reasons == []


@pytest.mark.asyncio
async def test_hard_stop_beats_max_hold(settings: HyperliquidSettings) -> None:
    """A position that is both stale and stopped out books the stop."""
    pm, reasons = await _pm_with_max_hold(settings, "60")
    await pm.register_position(_aged_position(120))
    await pm.on_price_update("BTC", Decimal("89"))  # through the 90 stop
    assert reasons == ["stop_loss"]


@pytest.mark.asyncio
async def test_trailing_beats_max_hold(settings: HyperliquidSettings) -> None:
    """With both armed the price-based exit wins, matching the simulator."""
    pm, reasons = await _pm_with_max_hold(settings, "60", trailing="0.01")
    await pm.register_position(_aged_position(120))
    await pm.on_price_update("BTC", Decimal("98"))  # 1% below the 100 peak
    assert reasons == ["trailing_stop"]


# --------------------------------------------------------------------------
# Live startup reconciliation
# --------------------------------------------------------------------------


def _live(settings: HyperliquidSettings) -> HyperliquidSettings:
    return settings.model_copy(update={"bot_dry_run": False})


def _ex(coin: str, szi: str, entry: str = "100") -> ExchangePosition:
    return ExchangePosition(
        coin=coin, signed_size=Decimal(szi), entry_px=Decimal(entry)
    )


async def _store_with(tmp_path, settings: HyperliquidSettings, *positions) -> OpenPositionStore:
    store_path = tmp_path / "open_positions.json"
    store = OpenPositionStore(
        settings.model_copy(update={"open_positions_path": str(store_path)}),
        path=store_path,
    )
    for p in positions:
        await store.upsert(p)
    return store


def _managed(coin: str = "BTC", side: SignalSide = SignalSide.LONG, size: str = "2") -> ManagedPosition:
    return ManagedPosition(
        coin=coin,
        side=side,
        entry_px=Decimal("100"),
        size=Decimal(size),
        stop_px=Decimal("98"),
        peak_px=Decimal("100"),
    )


@pytest.mark.asyncio
async def test_dry_run_skips_exchange_reconciliation(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """Dry-run has no real positions; the fetcher must not be consulted."""
    store = await _store_with(tmp_path, settings, _managed())
    fetcher = AsyncMock()
    pm = PositionManager(
        settings, kill_switch, open_position_store=store, exchange_positions=fetcher
    )
    recovered = await pm.recover_positions()
    fetcher.assert_not_awaited()
    assert [p.coin for p in recovered] == ["BTC"]


@pytest.mark.asyncio
async def test_live_without_fetcher_refuses_to_start(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    live = _live(settings)
    store = await _store_with(tmp_path, live, _managed())
    pm = PositionManager(live, kill_switch, open_position_store=store)
    with pytest.raises(PositionReconciliationError):
        await pm.recover_positions()


@pytest.mark.asyncio
async def test_live_fetch_failure_refuses_to_start(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """A failed read must not silently degrade into trading blind."""
    live = _live(settings)
    store = await _store_with(tmp_path, live, _managed())
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(side_effect=ConnectionError("boom")),
    )
    with pytest.raises(PositionReconciliationError):
        await pm.recover_positions()


@pytest.mark.asyncio
async def test_disk_position_absent_from_exchange_is_dropped(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """Closed/liquidated while down — keeping it would leave a phantom."""
    live = _live(settings)
    store = await _store_with(tmp_path, live, _managed())
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(return_value={}),
    )
    assert await pm.recover_positions() == []
    assert "BTC" not in pm.positions


@pytest.mark.asyncio
async def test_size_mismatch_trusts_the_exchange(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    live = _live(settings)
    store = await _store_with(tmp_path, live, _managed(size="2"))
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(return_value={"BTC": _ex("BTC", "5")}),
    )
    recovered = await pm.recover_positions()
    assert recovered[0].size == Decimal("5")


@pytest.mark.asyncio
async def test_side_mismatch_rebuilds_from_exchange(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """Disk says long, exchange says short: the exchange wins, stop recomputed."""
    live = _live(settings)
    store = await _store_with(tmp_path, live, _managed(side=SignalSide.LONG))
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(return_value={"BTC": _ex("BTC", "-3", "120")}),
    )
    pos = (await pm.recover_positions())[0]
    assert pos.side == SignalSide.SHORT
    assert pos.size == Decimal("3")
    assert pos.entry_px == Decimal("120")
    assert pos.stop_px > pos.entry_px  # short stop sits above entry


@pytest.mark.asyncio
async def test_untracked_exchange_position_is_adopted(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """An unmanaged live position has no stop; adopting beats ignoring."""
    live = _live(settings)
    store = await _store_with(tmp_path, live)
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(return_value={"SOL": _ex("SOL", "9", "50")}),
    )
    recovered = await pm.recover_positions()
    assert [p.coin for p in recovered] == ["SOL"]
    adopted = pm.positions["SOL"]
    assert adopted.size == Decimal("9")
    assert adopted.stop_px < adopted.entry_px  # long stop sits below entry
    assert "SOL" in await store.load()  # persisted so a later restart sees it


@pytest.mark.asyncio
async def test_matching_position_keeps_disk_stop(
    settings: HyperliquidSettings, kill_switch: KillSwitch, tmp_path
) -> None:
    """Agreement must not discard the trailing/break-even state on disk."""
    live = _live(settings)
    saved = _managed(size="2")
    saved.stop_px = Decimal("99.5")
    saved.break_even_armed = True
    store = await _store_with(tmp_path, live, saved)
    pm = PositionManager(
        live,
        kill_switch,
        open_position_store=store,
        exchange_positions=AsyncMock(return_value={"BTC": _ex("BTC", "2")}),
    )
    pos = (await pm.recover_positions())[0]
    assert pos.stop_px == Decimal("99.5")
    assert pos.break_even_armed is True


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


_ALO_REJECTED = {
    "status": "ok",
    "response": {
        "data": {
            "statuses": [
                {"error": "Order could not immediately match against any resting orders."}
            ]
        }
    },
}


@pytest.mark.asyncio
async def test_rejected_entry_does_not_register_a_position(
    settings: HyperliquidSettings,
) -> None:
    """A refused post-only entry must not create a position that does not exist.

    Hyperliquid returns this inside an HTTP 200, so nothing upstream raises. If
    the position were registered the bot would manage a phantom and later send a
    reduce-only close for it.
    """
    router = _build_router(settings)
    router.rest.place_order = AsyncMock(return_value=_ALO_REJECTED)

    result = await router.route_entry(_signal())

    assert result is None
    assert router.position_manager.positions == {}


@pytest.mark.asyncio
async def test_rejected_exit_keeps_the_position_under_management(
    settings: HyperliquidSettings,
) -> None:
    """A failed close must not drop the record.

    Forgetting it would leave a real position on the exchange with no stop
    attached and nothing watching it.
    """
    router = _build_router(settings)
    position = ManagedPosition(
        coin="BTC",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("98"),
        peak_px=Decimal("100"),
    )
    await router.position_manager.register_position(position)
    router.rest.place_order = AsyncMock(
        return_value={"status": "err", "response": "Insufficient margin"}
    )

    await router.route_exit(position, "stop_loss")

    assert "BTC" in router.position_manager.positions  # still managed, retries next tick


@pytest.mark.asyncio
async def test_accepted_exit_removes_the_position(settings: HyperliquidSettings) -> None:
    router = _build_router(settings)
    position = ManagedPosition(
        coin="BTC",
        side=SignalSide.LONG,
        entry_px=Decimal("100"),
        size=Decimal("1"),
        stop_px=Decimal("98"),
        peak_px=Decimal("100"),
    )
    await router.position_manager.register_position(position)
    router.rest.place_order = AsyncMock(
        return_value={"status": "ok", "response": {"data": {"statuses": [{"filled": {"oid": 1}}]}}}
    )

    await router.route_exit(position, "time_stop")

    assert "BTC" not in router.position_manager.positions


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
async def test_confidence_floor_rejects_weak_signal(settings: HyperliquidSettings) -> None:
    router = _build_router(settings.model_copy(update={"dvsla_min_confidence": Decimal("0.40")}))
    result = await router.route_entry(_signal(confidence=Decimal("0.39")))
    assert result is None
    router.rest.place_order.assert_not_called()
    assert "BTC" not in router.position_manager.positions


@pytest.mark.asyncio
async def test_confidence_floor_admits_signal_at_threshold(
    settings: HyperliquidSettings,
) -> None:
    router = _build_router(settings.model_copy(update={"dvsla_min_confidence": Decimal("0.40")}))
    result = await router.route_entry(_signal(confidence=Decimal("0.40")))
    assert result is not None
    router.rest.place_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_confidence_floor_of_zero_disables_the_gate(
    settings: HyperliquidSettings,
) -> None:
    # A floor of 0 must mean "no gate", not "reject everything at or below 0".
    router = _build_router(settings.model_copy(update={"dvsla_min_confidence": Decimal("0")}))
    result = await router.route_entry(_signal(confidence=Decimal("0")))
    assert result is not None
    router.rest.place_order.assert_awaited_once()


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




# --- simulated post-only lifecycle (dry-run only) ---------------------------
#
# Dry-run sends nothing, so nothing rejects and nothing has to be touched to
# fill. Every number the journal has produced carried both of those freebies.
# These cover the modelled versions.


def _maker_router(**overrides: object) -> OrderRouter:
    base: dict[str, object] = {
        "bot_dry_run": True,
        "maker_entry_enabled": True,
        "confidence_sizing_enabled": False,
        "atr_stop_enabled": False,
        "reversion_tp_enabled": False,
        "reentry_cooldown_seconds": 0,
        "dvsla_min_confidence": Decimal("0"),
        "max_hold_seconds": Decimal("60"),
        "mark_staleness_gap_seconds": Decimal("0"),  # no blackout guard in tests
    }
    base.update(overrides)
    return _build_router(HyperliquidSettings.from_env().model_copy(update=base))


@pytest.mark.asyncio
async def test_post_only_entry_rests_instead_of_opening_a_position() -> None:
    router = _maker_router()

    await router.route_entry(_signal(symbol="BTC"))

    assert router.position_manager.positions == {}, "the order has not filled yet"
    assert "BTC" in router._pending


@pytest.mark.asyncio
async def test_post_only_entry_is_rejected_when_it_would_cross_on_arrival() -> None:
    # Long resting at 50000; by the time the order lands the market is at 49900,
    # so the bid is through it and a real exchange refuses the order outright.
    router = _maker_router()
    await router.route_entry(_signal(side=SignalSide.LONG, symbol="BTC"))

    await router.on_mark("BTC", Decimal("49900"))

    assert router._pending == {}
    assert router.position_manager.positions == {}


@pytest.mark.asyncio
async def test_post_only_entry_fills_when_price_comes_back_to_it() -> None:
    router = _maker_router()
    await router.route_entry(_signal(side=SignalSide.LONG, symbol="BTC"))

    await router.on_mark("BTC", Decimal("50100"))  # arrival: rests below the market
    assert "BTC" in router._pending
    assert router.position_manager.positions == {}

    await router.on_mark("BTC", Decimal("50000"))  # price returns and touches it

    assert router._pending == {}
    position = router.position_manager.positions["BTC"]
    # Filled at the limit, not at whatever the mark happened to be.
    assert position.entry_px == Decimal("50000")


@pytest.mark.asyncio
async def test_short_post_only_entry_is_rejected_when_market_is_above_it() -> None:
    router = _maker_router()
    await router.route_entry(_signal(side=SignalSide.SHORT, symbol="BTC"))

    await router.on_mark("BTC", Decimal("50100"))  # ask through the offer

    assert router._pending == {}
    assert router.position_manager.positions == {}


@pytest.mark.asyncio
async def test_resting_entry_expires_unfilled_after_the_hold_window() -> None:
    router = _maker_router()
    await router.route_entry(_signal(side=SignalSide.LONG, symbol="BTC"))
    await router.on_mark("BTC", Decimal("50100"))  # arrives and rests

    router._pending["BTC"].placed_at -= timedelta(seconds=61)
    await router.on_mark("BTC", Decimal("50100"))

    assert router._pending == {}
    assert router.position_manager.positions == {}


@pytest.mark.asyncio
async def test_second_signal_is_skipped_while_an_entry_is_resting() -> None:
    router = _maker_router()
    await router.route_entry(_signal(symbol="BTC"))
    router.rest.place_order.reset_mock()

    result = await router.route_entry(_signal(symbol="BTC"))

    assert result is None
    router.rest.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_mode_does_not_simulate_the_exchange() -> None:
    # Live, the exchange rejects and fills for real and reports back; modelling
    # it here as well would double-count and hide real fills behind a guess.
    router = _maker_router(bot_dry_run=False, wallet_address="0x" + "1" * 40)

    await router.route_entry(_signal(symbol="BTC"))

    assert router._pending == {}
    assert "BTC" in router.position_manager.positions


@pytest.mark.asyncio
async def test_taker_dry_run_still_opens_immediately() -> None:
    # Nothing to model without a post-only order: an IOC either fills or does
    # not, and the response says which.
    router = _maker_router(maker_entry_enabled=False)

    await router.route_entry(_signal(symbol="BTC"))

    assert router._pending == {}
    assert "BTC" in router.position_manager.positions
