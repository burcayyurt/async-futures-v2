from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from src.core.config import HyperliquidSettings
from src.core.telegram import TelegramNotifier
from src.exchange.hyperliquid_rest import HyperliquidRestClient, OrderRequest
from src.execution.position_manager import ManagedPosition, PositionManager
from src.risk.kill_switch import ExecutionLockedError, KillSwitch
from src.risk.margin_manager import MarginManager, MarginSafetyError, MarginSnapshot
from src.strategy.momentum_oi import SignalSide, TradeSignal

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
    ) -> None:
        self.settings = settings
        self.rest = rest
        self.kill_switch = kill_switch
        self.margin_manager = margin_manager
        self.position_manager = position_manager
        self._telegram = telegram
        self.market_slippage_pct = market_slippage_pct
        self.position_manager.bind_exit_handler(self.route_exit)

    @staticmethod
    def _slippage_price(mark_px: Decimal, is_buy: bool, slippage_pct: Decimal) -> Decimal:
        if is_buy:
            return mark_px * (Decimal("1") + slippage_pct)
        return mark_px * (Decimal("1") - slippage_pct)

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

        if coin in self.position_manager.positions:
            logger.info("Entry skipped for %s: position already open", coin)
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

        is_buy = signal.side == SignalSide.LONG
        limit_px = self._slippage_price(signal.entry_mark_price, is_buy, self.market_slippage_pct)
        order = OrderRequest(
            coin=coin,
            is_buy=is_buy,
            sz=str(size),
            limit_px=str(limit_px),
            reduce_only=False,
            tif="Ioc",
        )
        result = await self.rest.place_order(order)

        position = ManagedPosition(
            coin=coin,
            side=signal.side,
            entry_px=signal.entry_mark_price,
            size=Decimal(str(size)),
            stop_px=PositionManager.initial_stop_price(signal.entry_mark_price, signal.side),
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
        await self.position_manager.remove_position(coin)
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
