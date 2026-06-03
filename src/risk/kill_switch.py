from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone
from decimal import Decimal

from src.core.config import HyperliquidSettings
from src.core.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

DRAWDOWN_DEBOUNCE_SECONDS = 3


class ExecutionLockedError(RuntimeError):
    """Raised when trading is blocked by the kill switch."""


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class KillSwitch:
    """Mark-price-aware daily drawdown kill switch with debounced trip confirmation."""

    def __init__(
        self,
        settings: HyperliquidSettings,
        *,
        telegram: TelegramNotifier | None = None,
    ) -> None:
        self.settings = settings
        self._telegram = telegram
        self._lock = asyncio.Lock()
        self._drawdown_confirm_lock = asyncio.Lock()
        self._tripped = False
        self._trip_reason: str | None = None
        self._mark_prices: dict[str, Decimal] = {}
        self._daily_utc_date: date | None = None
        self._daily_start_balance: Decimal | None = None
        self._daily_high_water_mark: Decimal | None = None

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str | None:
        return self._trip_reason

    @property
    def mark_prices(self) -> dict[str, Decimal]:
        return dict(self._mark_prices)

    @staticmethod
    def _drawdown_pct(reference_balance: Decimal, equity: Decimal) -> Decimal:
        if reference_balance <= 0:
            return Decimal("0")
        return ((reference_balance - equity) / reference_balance) * Decimal("100")

    async def update_mark_price(self, coin: str, mark_px: Decimal) -> None:
        if mark_px <= 0:
            raise ValueError("mark_px must be positive")
        self._mark_prices[coin.strip().upper()] = mark_px

    async def _reset_daily_state(self, equity: Decimal, *, auto_unlock: bool = True) -> None:
        async with self._lock:
            was_tripped = self._tripped
            today = _utc_today()
            self._daily_utc_date = today
            self._daily_start_balance = equity
            self._daily_high_water_mark = equity
            if auto_unlock and was_tripped:
                self._tripped = False
                self._trip_reason = None

        if auto_unlock and was_tripped:
            logger.warning(
                "Kill switch unlocked for new UTC day %s (reference equity=%s)",
                today.isoformat(),
                equity,
            )
        else:
            logger.info(
                "Daily drawdown reference set for UTC %s: start=%s high_water=%s",
                today.isoformat(),
                equity,
                equity,
            )

    async def _update_high_water_mark(self, current_equity: Decimal) -> tuple[Decimal, Decimal]:
        today = _utc_today()
        async with self._lock:
            if self._daily_utc_date != today or self._daily_high_water_mark is None:
                self._daily_utc_date = today
                self._daily_start_balance = current_equity
                self._daily_high_water_mark = current_equity
            elif current_equity > self._daily_high_water_mark:
                self._daily_high_water_mark = current_equity

            reference = self._daily_high_water_mark
            day_start = self._daily_start_balance or current_equity
        return reference, day_start

    async def _confirm_drawdown_trip(
        self,
        *,
        initial_loss_pct: Decimal,
        reference_balance: Decimal,
        daily_start_balance: Decimal,
        mark_price: Decimal,
        recheck_equity: Callable[[Decimal], Awaitable[Decimal]] | None,
    ) -> bool:
        async with self._drawdown_confirm_lock:
            if self._tripped:
                return True

            logger.warning(
                "Drawdown threshold exceeded (%.2f%%) at mark_price=%s, confirming in %ss...",
                initial_loss_pct,
                mark_price,
                DRAWDOWN_DEBOUNCE_SECONDS,
            )
            await asyncio.sleep(DRAWDOWN_DEBOUNCE_SECONDS)

            if self._tripped:
                return True

            if recheck_equity is None:
                logger.warning("Drawdown confirm skipped: no recheck_equity callback provided")
                return False

            confirmed_equity = await recheck_equity(mark_price)
            confirmed_loss_pct = self._drawdown_pct(reference_balance, confirmed_equity)

            if confirmed_loss_pct >= self.settings.max_drawdown_pct:
                reason = (
                    f"Daily drawdown {confirmed_loss_pct:.2f}% exceeded "
                    f"{self.settings.max_drawdown_pct}% "
                    f"(day start={daily_start_balance}, day high={reference_balance}, "
                    f"mark_price={mark_price})"
                )
                await self.trip(reason)
                return True

            logger.info(
                "Drawdown false alarm dismissed after %ss confirm "
                "(equity recovered to %s at mark_price=%s)",
                DRAWDOWN_DEBOUNCE_SECONDS,
                confirmed_equity,
                mark_price,
            )
            return False

    async def check_drawdown(
        self,
        current_equity: Decimal,
        mark_price: Decimal,
        *,
        coin: str | None = None,
        recheck_equity: Callable[[Decimal], Awaitable[Decimal]] | None = None,
    ) -> bool:
        if mark_price <= 0:
            raise ValueError("mark_price must be positive for drawdown checks")

        if coin:
            await self.update_mark_price(coin, mark_price)

        today = _utc_today()
        if self._daily_utc_date != today:
            await self._reset_daily_state(current_equity, auto_unlock=True)

        reference_balance, daily_start_balance = await self._update_high_water_mark(current_equity)
        loss_pct = self._drawdown_pct(reference_balance, current_equity)

        if loss_pct < self.settings.max_drawdown_pct:
            return False

        if recheck_equity is None:
            reason = (
                f"Daily drawdown {loss_pct:.2f}% exceeded "
                f"{self.settings.max_drawdown_pct}% "
                f"(day start={daily_start_balance}, day high={reference_balance}, "
                f"mark_price={mark_price})"
            )
            await self.trip(reason)
            return True

        return await self._confirm_drawdown_trip(
            initial_loss_pct=loss_pct,
            reference_balance=reference_balance,
            daily_start_balance=daily_start_balance,
            mark_price=mark_price,
            recheck_equity=recheck_equity,
        )

    async def trip(self, reason: str) -> None:
        async with self._lock:
            if self._tripped:
                return
            self._tripped = True
            self._trip_reason = reason
        logger.critical("KILL SWITCH TRIPPED! Reason: %s", reason)
        if self._telegram is not None:
            self._telegram.notify_kill_switch()

    async def reset(self) -> None:
        async with self._lock:
            was_tripped = self._tripped
            self._tripped = False
            self._trip_reason = None
        if was_tripped:
            logger.warning("Kill switch manually reset")
        else:
            logger.info("Kill switch reset requested but switch was already open")

    async def assert_can_trade(self) -> None:
        if self._tripped:
            reason = self._trip_reason or "Kill switch is active"
            raise ExecutionLockedError(reason)
