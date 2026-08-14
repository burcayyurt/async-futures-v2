from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.config import HyperliquidSettings
from src.core.telegram import TelegramNotifier
from src.exchange.hyperliquid_rest import (
    HyperliquidRestClient,
    OrderRequest,
    order_rejection,
)
from src.execution.position_manager import ManagedPosition, PositionManager
from src.risk.kill_switch import ExecutionLockedError, KillSwitch
from src.risk.margin_manager import MarginManager, MarginSafetyError, MarginSnapshot
from src.strategy.signals import SignalSide, TradeSignal

logger = logging.getLogger(__name__)

DRY_RUN_FALLBACK_COLLATERAL = Decimal("1000")


class OrderRouter:
    """Routes strategy signals to Hyperliquid and manages position lifecycle."""

    def __init__(
        self,
        settings: HyperliquidSettings,
        rest: HyperliquidRestClient,
        kill_switch: KillSwitch,
        margin_manager: MarginManager,
        position_manager: PositionManager,
        *,
        telegram: TelegramNotifier | None = None,
        market_slippage_pct: Decimal = Decimal("0.01"),
        feed_health: Callable[[], float | None] | None = None,
    ) -> None:
        self.settings = settings
        self.rest = rest
        self.kill_switch = kill_switch
        self.margin_manager = margin_manager
        self.position_manager = position_manager
        self._telegram = telegram
        self.market_slippage_pct = market_slippage_pct
        # Returns seconds since the market feed (re)connected, or None if it is
        # currently down. Used to gate entries during the post-reconnect warm-up.
        self._feed_health = feed_health
        self._last_exit_at: dict[str, datetime] = {}
        self.position_manager.bind_exit_handler(self.route_exit)

    def _feed_warming_up(self, coin: str) -> bool:
        if self._feed_health is None:
            return False
        since = self._feed_health()
        if since is None:
            logger.warning("Entry rejected for %s: market feed is down (blackout)", coin)
            return True
        cooldown = self.settings.ws_reconnect_entry_cooldown_seconds
        if cooldown > 0 and since < cooldown:
            logger.info(
                "Entry skipped for %s: feed warm-up after reconnect (%.1fs < %ss)",
                coin,
                since,
                cooldown,
            )
            return True
        return False

    @staticmethod
    def _slippage_price(mark_px: Decimal, is_buy: bool, slippage_pct: Decimal) -> Decimal:
        if is_buy:
            return mark_px * (Decimal("1") + slippage_pct)
        return mark_px * (Decimal("1") - slippage_pct)

    def _in_cooldown(self, coin: str, now: datetime) -> bool:
        cooldown = self.settings.reentry_cooldown_seconds
        if cooldown <= 0:
            return False
        last = self._last_exit_at.get(coin)
        if last is None:
            return False
        return (now - last).total_seconds() < cooldown

    def _confidence_scaled_size(self, size: float, confidence: Decimal) -> float:
        if not self.settings.confidence_sizing_enabled:
            return size
        floor = self.settings.confidence_size_floor
        conf = max(Decimal("0"), min(Decimal("1"), confidence))
        factor = floor + (Decimal("1") - floor) * conf
        return float(Decimal(str(size)) * factor)

    async def _fetch_collateral_snapshot(self) -> MarginSnapshot:
        try:
            return await self.margin_manager.fetch_margin_snapshot()
        except Exception:
            if self.settings.bot_dry_run:
                logger.warning("Using dry-run fallback collateral snapshot")
                return MarginSnapshot(
                    equity=DRY_RUN_FALLBACK_COLLATERAL,
                    used_margin=Decimal("0"),
                    available_margin=DRY_RUN_FALLBACK_COLLATERAL,
                    leverage=self.settings.leverage,
                )
            raise

    async def consume_signals(self, queue: asyncio.Queue[TradeSignal]) -> None:
        while True:
            signal = await queue.get()
            try:
                await self.route_entry(signal)
            except Exception:
                logger.exception("Failed routing signal for %s", signal.symbol)

    async def route_entry(self, signal: TradeSignal) -> dict[str, Any] | None:
        coin = signal.symbol.strip().upper()
        try:
            await self.kill_switch.assert_can_trade()
        except ExecutionLockedError as exc:
            logger.warning("Entry rejected for %s: %s", coin, exc)
            return None

        if self._feed_warming_up(coin):
            return None

        if coin in self.position_manager.positions:
            logger.info("Entry skipped for %s: position already open", coin)
            return None

        now = datetime.now(timezone.utc)
        if self._in_cooldown(coin, now):
            logger.info("Entry skipped for %s: re-entry cooldown active", coin)
            return None

        snapshot = await self._fetch_collateral_snapshot()
        try:
            size = self.margin_manager.calculate_position_size(
                free_collateral=float(snapshot.available_margin),
                mark_price=float(signal.entry_mark_price),
                risk_pct=float(self.settings.trade_risk_pct),
                leverage=self.settings.leverage,
            )
        except MarginSafetyError as exc:
            logger.warning("Entry rejected for %s: %s", coin, exc)
            return None

        size = self._confidence_scaled_size(size, signal.confidence)
        if size <= 0:
            logger.info("Entry skipped for %s: confidence-scaled size is zero", coin)
            return None

        is_buy = signal.side == SignalSide.LONG
        if self.settings.maker_entry_enabled:
            # Passive post-only maker entry at the signal price (no spread cross).
            limit_px = signal.entry_mark_price
            tif: str = "Alo"
        else:
            limit_px = self._slippage_price(signal.entry_mark_price, is_buy, self.market_slippage_pct)
            tif = "Ioc"
        order = OrderRequest(
            coin=coin,
            is_buy=is_buy,
            sz=str(size),
            limit_px=str(limit_px),
            reduce_only=False,
            tif=tif,
        )
        result = await self.rest.place_order(order)

        rejection = order_rejection(result)
        if rejection is not None:
            # Registering here would create a position the exchange never opened,
            # which the bot would then try to stop out and reduce-only close.
            logger.warning("Entry rejected by exchange for %s: %s", coin, rejection)
            return None

        position = ManagedPosition(
            coin=coin,
            side=signal.side,
            entry_px=signal.entry_mark_price,
            size=Decimal(str(size)),
            stop_px=self.position_manager.compute_initial_stop(
                coin, signal.entry_mark_price, signal.side
            ),
            peak_px=signal.entry_mark_price,
        )
        await self.position_manager.register_position(position)
        if self._telegram is not None:
            self._telegram.notify_entry(coin, signal.side, signal.entry_mark_price, Decimal(str(size)))
        logger.info(
            "Entry routed for %s %s size=%s mark=%s result=%s",
            coin,
            signal.side.value,
            size,
            signal.entry_mark_price,
            result.get("status", result),
        )
        return result

    async def route_exit(self, position: ManagedPosition, reason: str) -> dict[str, Any] | None:
        coin = position.coin.strip().upper()
        if coin not in self.position_manager.positions:
            logger.debug("Exit skipped for %s: position not tracked", coin)
            return None

        is_buy = position.side == SignalSide.SHORT
        mark_px = position.peak_px or position.entry_px
        limit_px = self._slippage_price(mark_px, is_buy, self.market_slippage_pct)
        order = OrderRequest(
            coin=coin,
            is_buy=is_buy,
            sz=str(position.size),
            limit_px=str(limit_px),
            reduce_only=True,
            tif="Ioc",
        )
        result = await self.rest.place_order(order)

        rejection = order_rejection(result)
        if rejection is not None:
            # Dropping the record here would leave a real, unmanaged position on
            # the exchange with no stop attached. Keep it so the next mark can
            # retry the exit.
            logger.error(
                "Exit rejected by exchange for %s (reason=%s): %s; position stays open",
                coin,
                reason,
                rejection,
            )
            return result

        await self.position_manager.remove_position(coin)
        self._last_exit_at[coin] = datetime.now(timezone.utc)
        logger.info("Exit routed for %s reason=%s result=%s", coin, reason, result.get("status", result))
        return result

    async def route_stop_update(
        self,
        position: ManagedPosition,
        new_stop_px: str,
    ) -> dict[str, Any]:
        position.stop_px = Decimal(new_stop_px)
        logger.debug("Updated internal stop for %s to %s", position.coin, new_stop_px)
        return {"status": "stop_updated", "coin": position.coin, "stop_px": new_stop_px}
