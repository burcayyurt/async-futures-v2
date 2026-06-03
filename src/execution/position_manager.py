from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from src.core.config import HyperliquidSettings
from src.core.open_position_store import OpenPositionStore
from src.core.telegram import TelegramNotifier
from src.core.trade_journal import TradeJournal, calc_roe_pct
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, MarketEvent
from src.risk.kill_switch import KillSwitch
from src.strategy.momentum_oi import SignalSide

logger = logging.getLogger(__name__)

FEE_BUFFER_PCT = Decimal("0.001")
INITIAL_STOP_PCT = Decimal("0.02")
PEAK_PERSIST_DEBOUNCE_SECONDS = 5.0

ExitHandler = Callable[["ManagedPosition", str], Awaitable[None]]


@dataclass(slots=True)
class ManagedPosition:
    coin: str
    side: SignalSide
    entry_px: Decimal
    size: Decimal
    stop_px: Decimal
    break_even_armed: bool = False
    peak_px: Decimal | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PositionManager:
    """Mark-price-based trailing stop, break-even, and kill-switch panic exit."""

    def __init__(
        self,
        settings: HyperliquidSettings,
        kill_switch: KillSwitch,
        *,
        telegram: TelegramNotifier | None = None,
        trade_journal: TradeJournal | None = None,
        open_position_store: OpenPositionStore | None = None,
    ) -> None:
        self.settings = settings
        self.kill_switch = kill_switch
        self._telegram = telegram
        self._journal = trade_journal
        self._open_store = open_position_store
        self._positions: dict[str, ManagedPosition] = {}
        self._exit_handler: ExitHandler | None = None
        self._panic_in_progress = False
        self._last_peak_persist: dict[str, datetime] = {}

    @property
    def positions(self) -> dict[str, ManagedPosition]:
        return dict(self._positions)

    def bind_exit_handler(self, handler: ExitHandler) -> None:
        self._exit_handler = handler

    async def recover_positions(self) -> list[ManagedPosition]:
        if self._open_store is None:
            return []

        saved = await self._open_store.load()
        if not saved:
            return []

        allowed = {symbol.strip().upper() for symbol in self.settings.symbols}
        recovered: list[ManagedPosition] = []

        for coin, position in saved.items():
            if coin not in allowed:
                logger.warning("Removing stale open position for %s (not in BOT_SYMBOLS)", coin)
                await self._open_store.remove(coin)
                continue

            if not self.settings.bot_dry_run:
                logger.info(
                    "Recovered position %s from disk; exchange reconciliation is Phase 2 for live mode",
                    coin,
                )

            self._positions[coin] = position
            recovered.append(position)
            logger.info(
                "Recovered position %s %s size=%s entry=%s stop=%s",
                coin,
                position.side.value,
                position.size,
                position.entry_px,
                position.stop_px,
            )

        return recovered

    async def register_position(self, position: ManagedPosition) -> None:
        coin = position.coin.strip().upper()
        if position.peak_px is None:
            position.peak_px = position.entry_px
        self._positions[coin] = position
        if self._open_store is not None:
            await self._open_store.upsert(position)
        logger.info(
            "Registered position %s %s size=%s entry=%s stop=%s",
            coin,
            position.side.value,
            position.size,
            position.entry_px,
            position.stop_px,
        )

    async def remove_position(self, coin: str) -> None:
        normalized = coin.strip().upper()
        self._positions.pop(normalized, None)
        self._last_peak_persist.pop(normalized, None)
        if self._open_store is not None:
            await self._open_store.remove(normalized)

    async def _maybe_persist_position(self, position: ManagedPosition, now: datetime) -> None:
        if self._open_store is None:
            return
        coin = position.coin.strip().upper()
        last = self._last_peak_persist.get(coin)
        if last is not None and (now - last).total_seconds() < PEAK_PERSIST_DEBOUNCE_SECONDS:
            return
        self._last_peak_persist[coin] = now
        await self._open_store.upsert(position)

    def _gain_pct(self, position: ManagedPosition, mark_px: Decimal) -> Decimal:
        if position.entry_px <= 0:
            return Decimal("0")
        if position.side == SignalSide.LONG:
            return ((mark_px - position.entry_px) / position.entry_px) * Decimal("100")
        return ((position.entry_px - mark_px) / position.entry_px) * Decimal("100")

    def _update_peak(self, position: ManagedPosition, mark_px: Decimal) -> None:
        if position.peak_px is None:
            position.peak_px = mark_px
            return
        if position.side == SignalSide.LONG:
            position.peak_px = max(position.peak_px, mark_px)
        else:
            position.peak_px = min(position.peak_px, mark_px)

    async def maybe_move_to_break_even(
        self,
        position: ManagedPosition,
        mark_px: Decimal,
    ) -> Decimal | None:
        if self.settings.break_even_trigger_pct <= 0:
            return None
        if position.break_even_armed:
            return position.stop_px

        gain_pct = self._gain_pct(position, mark_px)
        if gain_pct < self.settings.break_even_trigger_pct:
            return None

        if position.side == SignalSide.LONG:
            new_stop = position.entry_px * (Decimal("1") + FEE_BUFFER_PCT)
            position.stop_px = max(position.stop_px, new_stop)
        else:
            new_stop = position.entry_px * (Decimal("1") - FEE_BUFFER_PCT)
            position.stop_px = min(position.stop_px, new_stop)

        position.break_even_armed = True
        if self._open_store is not None:
            await self._open_store.upsert(position)
        logger.info(
            "Break-even armed for %s at stop=%s (mark=%s gain=%.2f%%)",
            position.coin,
            position.stop_px,
            mark_px,
            gain_pct,
        )
        return position.stop_px

    async def maybe_trail_stop(
        self,
        position: ManagedPosition,
        mark_px: Decimal,
    ) -> Decimal | None:
        if position.peak_px is None or position.peak_px <= 0:
            return None

        callback = self.settings.trailing_callback_pct
        if position.side == SignalSide.LONG:
            trigger_px = position.peak_px * (Decimal("1") - callback)
            if mark_px <= trigger_px:
                return trigger_px
        else:
            trigger_px = position.peak_px * (Decimal("1") + callback)
            if mark_px >= trigger_px:
                return trigger_px
        return None

    def _hit_hard_stop(self, position: ManagedPosition, mark_px: Decimal) -> bool:
        if position.side == SignalSide.LONG:
            return mark_px <= position.stop_px
        return mark_px >= position.stop_px

    @staticmethod
    def _calc_pnl_usd(position: ManagedPosition, exit_px: Decimal) -> Decimal:
        if position.side == SignalSide.LONG:
            return (exit_px - position.entry_px) * position.size
        return (position.entry_px - exit_px) * position.size

    async def _request_exit(
        self,
        position: ManagedPosition,
        reason: str,
        *,
        exit_px: Decimal,
    ) -> None:
        pnl = self._calc_pnl_usd(position, exit_px)
        roe_pct = calc_roe_pct(
            side=position.side,
            entry_px=position.entry_px,
            exit_px=exit_px,
            leverage=self.settings.leverage,
        )
        if self._journal is not None:
            self._journal.record_closed_trade(position, exit_px, pnl, roe_pct, reason)
        if self._telegram is not None:
            self._telegram.notify_exit(
                position.coin,
                exit_px,
                pnl,
                roe_pct=roe_pct,
                exit_reason=reason,
                side=position.side,
            )
        if self._exit_handler is None:
            logger.error("No exit handler bound; cannot close %s", position.coin)
            return
        await self._exit_handler(position, reason)

    async def panic_close_all(self) -> None:
        if self._panic_in_progress or not self._positions:
            return
        self._panic_in_progress = True
        try:
            positions = list(self._positions.values())
            logger.critical("Panic closing %d open position(s)", len(positions))
            for position in positions:
                exit_px = self.kill_switch.mark_prices.get(position.coin)
                if exit_px is None:
                    exit_px = position.peak_px or position.entry_px
                await self._request_exit(position, "kill_switch", exit_px=exit_px)
        finally:
            self._panic_in_progress = False

    async def on_price_update(self, coin: str, mark_px: Decimal) -> None:
        if self.kill_switch.is_tripped:
            await self.panic_close_all()
            return

        normalized = coin.strip().upper()
        position = self._positions.get(normalized)
        if position is None:
            return

        previous_peak = position.peak_px
        self._update_peak(position, mark_px)
        if position.peak_px != previous_peak:
            await self._maybe_persist_position(position, datetime.now(timezone.utc))

        await self.maybe_move_to_break_even(position, mark_px)

        if self._hit_hard_stop(position, mark_px):
            await self._request_exit(position, "stop_loss", exit_px=mark_px)
            return

        trail_trigger = await self.maybe_trail_stop(position, mark_px)
        if trail_trigger is not None:
            await self._request_exit(position, "trailing_stop", exit_px=mark_px)

    async def consume_market_events(self, queue: asyncio.Queue[MarketEvent]) -> None:
        while True:
            event = await queue.get()
            try:
                if event.kind != EventKind.ASSET_CTX or event.coin is None:
                    continue
                if not isinstance(event.payload, AssetCtxPayload):
                    continue
                await self.on_price_update(event.coin, event.payload.mark_px)
            except Exception:
                logger.exception("Position manager failed processing market event")

    @staticmethod
    def initial_stop_price(entry_px: Decimal, side: SignalSide) -> Decimal:
        if side == SignalSide.LONG:
            return entry_px * (Decimal("1") - INITIAL_STOP_PCT)
        return entry_px * (Decimal("1") + INITIAL_STOP_PCT)
