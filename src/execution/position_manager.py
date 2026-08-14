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
from src.exchange.hyperliquid_rest import ExchangePosition
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, MarketEvent
from src.risk.kill_switch import KillSwitch
from src.strategy.microstructure import RollingStats
from src.strategy.signals import SignalSide

logger = logging.getLogger(__name__)

FEE_BUFFER_PCT = Decimal("0.001")
INITIAL_STOP_PCT = Decimal("0.02")
PEAK_PERSIST_DEBOUNCE_SECONDS = 5.0
# Relative size difference below which disk and exchange are treated as agreeing.
SIZE_RECONCILE_TOLERANCE = Decimal("0.001")

ExitHandler = Callable[["ManagedPosition", str], Awaitable[None]]
ExchangePositionFetcher = Callable[[], Awaitable[dict[str, ExchangePosition]]]


class PositionReconciliationError(RuntimeError):
    """Raised when live position state cannot be established at startup.

    Trading on an unknown position set is how a restart turns into a doubled or
    unhedged position, so the bot refuses to start rather than guess.
    """


class _MarkTracker:
    """Per-symbol rolling mark-price stats for ATR-stop and reversion-TP.

    Given the FAZ-0-Lite feed (no L2, asset_ctx mark prints only) we approximate
    a volatility band ("ATR proxy") from the rolling mean absolute mark return,
    and a reversion target from the rolling mean mark price. True volume-ATR is
    computed in the strategy/backtest where the trade tape is available.
    """

    __slots__ = ("_prices", "_abs_returns", "_prev")

    def __init__(self, atr_window: int, reversion_window: int) -> None:
        self._prices = RollingStats(reversion_window)
        self._abs_returns = RollingStats(atr_window)
        self._prev: Decimal | None = None

    def update(self, mark_px: Decimal) -> None:
        if mark_px <= 0:
            return
        self._prices.update(float(mark_px))
        if self._prev is not None and self._prev > 0:
            self._abs_returns.update(abs(float((mark_px - self._prev) / self._prev)))
        self._prev = mark_px

    def resync(self, mark_px: Decimal) -> None:
        """Re-anchor the price after a feed gap without recording the jump.

        The return across a blackout spans the whole gap and would otherwise
        blow up the ATR proxy; this resets the reference so the next tick yields
        a clean single-tick return.
        """
        if mark_px > 0:
            self._prev = mark_px

    @property
    def reversion_target(self) -> Decimal | None:
        if self._prices.count < 2:
            return None
        return Decimal(str(self._prices.mean))

    @property
    def atr_pct(self) -> Decimal | None:
        if self._abs_returns.count < 2:
            return None
        return Decimal(str(self._abs_returns.mean))


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
        exchange_positions: ExchangePositionFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.kill_switch = kill_switch
        self._telegram = telegram
        self._journal = trade_journal
        self._open_store = open_position_store
        self._exchange_positions = exchange_positions
        self._positions: dict[str, ManagedPosition] = {}
        self._exit_handler: ExitHandler | None = None
        self._panic_in_progress = False
        self._last_peak_persist: dict[str, datetime] = {}
        self._mark_trackers: dict[str, _MarkTracker] = {}
        self._last_mark_ts: dict[str, datetime] = {}

    @property
    def positions(self) -> dict[str, ManagedPosition]:
        return dict(self._positions)

    def bind_exit_handler(self, handler: ExitHandler) -> None:
        self._exit_handler = handler

    async def _fetch_exchange_positions(self) -> dict[str, ExchangePosition] | None:
        """Exchange truth for reconciliation, or ``None`` when not applicable.

        Dry-run has no wallet and no real positions, so the disk record is the
        only truth there. In live mode a missing or failing fetcher is fatal:
        starting blind risks re-opening a position that is already on.
        """

        if self.settings.bot_dry_run:
            return None
        if self._exchange_positions is None:
            raise PositionReconciliationError(
                "Live mode requires an exchange position fetcher; refusing to start "
                "without reconciliation."
            )
        try:
            return await self._exchange_positions()
        except Exception as exc:  # noqa: BLE001 - re-raised as a fatal startup error
            raise PositionReconciliationError(
                f"Could not read positions from the exchange: {exc}"
            ) from exc

    async def recover_positions(self) -> list[ManagedPosition]:
        if self._open_store is None:
            return []

        exchange = await self._fetch_exchange_positions()
        saved = await self._open_store.load()
        if not saved and not exchange:
            return []

        allowed = {symbol.strip().upper() for symbol in self.settings.symbols}
        recovered: list[ManagedPosition] = []

        for coin, position in saved.items():
            if coin not in allowed:
                logger.warning("Removing stale open position for %s (not in BOT_SYMBOLS)", coin)
                await self._open_store.remove(coin)
                continue

            if exchange is not None:
                live = exchange.get(coin)
                if live is None:
                    # Closed (or liquidated) while the bot was down. Keeping it
                    # would leave a phantom the exit handler can never fill.
                    logger.warning(
                        "Dropping %s: on disk but not open at the exchange", coin
                    )
                    await self._open_store.remove(coin)
                    continue
                position = self._reconcile(coin, position, live)

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

        if exchange is not None:
            recovered.extend(await self._adopt_orphans(exchange, saved, allowed))

        return recovered

    def _reconcile(
        self, coin: str, position: ManagedPosition, live: ExchangePosition
    ) -> ManagedPosition:
        """Return the disk position corrected against exchange truth."""

        live_side = SignalSide.LONG if live.is_long else SignalSide.SHORT
        if live_side != position.side:
            logger.warning(
                "Side mismatch for %s: disk=%s exchange=%s; trusting the exchange",
                coin,
                position.side.value,
                live_side.value,
            )
            entry_px = live.entry_px or position.entry_px
            return ManagedPosition(
                coin=coin,
                side=live_side,
                entry_px=entry_px,
                size=live.abs_size,
                stop_px=self.compute_initial_stop(coin, entry_px, live_side),
                peak_px=entry_px,
                opened_at=position.opened_at,
            )

        if position.size > 0:
            drift = abs(live.abs_size - position.size) / position.size
            if drift > SIZE_RECONCILE_TOLERANCE:
                logger.warning(
                    "Size mismatch for %s: disk=%s exchange=%s; trusting the exchange",
                    coin,
                    position.size,
                    live.abs_size,
                )
                position.size = live.abs_size
        return position

    async def _adopt_orphans(
        self,
        exchange: dict[str, ExchangePosition],
        saved: dict[str, ManagedPosition],
        allowed: set[str],
    ) -> list[ManagedPosition]:
        """Take over exchange positions the bot has no record of.

        An unmanaged live position has no stop attached, so leaving it alone is
        the worst option; adopting it at least puts it under stop management.
        Symbols outside ``BOT_SYMBOLS`` are adopted too — the risk is real money
        regardless of whether the bot would trade that symbol today.
        """

        adopted: list[ManagedPosition] = []
        for coin, live in exchange.items():
            if coin in saved or coin in self._positions:
                continue
            side = SignalSide.LONG if live.is_long else SignalSide.SHORT
            entry_px = live.entry_px
            if entry_px <= 0:
                logger.error(
                    "Orphan position %s has no usable entry price; leaving unmanaged", coin
                )
                continue
            position = ManagedPosition(
                coin=coin,
                side=side,
                entry_px=entry_px,
                size=live.abs_size,
                stop_px=self.compute_initial_stop(coin, entry_px, side),
                peak_px=entry_px,
            )
            self._positions[coin] = position
            adopted.append(position)
            if self._open_store is not None:
                await self._open_store.upsert(position)
            logger.warning(
                "Adopted untracked exchange position %s %s size=%s entry=%s stop=%s%s",
                coin,
                side.value,
                position.size,
                entry_px,
                position.stop_px,
                "" if coin in allowed else " (not in BOT_SYMBOLS)",
            )
        return adopted

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

    def _tracker(self, coin: str) -> _MarkTracker:
        normalized = coin.strip().upper()
        tracker = self._mark_trackers.get(normalized)
        if tracker is None:
            tracker = _MarkTracker(self.settings.atr_window, self.settings.reversion_window)
            self._mark_trackers[normalized] = tracker
        return tracker

    def compute_initial_stop(self, coin: str, entry_px: Decimal, side: SignalSide) -> Decimal:
        """Volatility-aware initial stop; falls back to the fixed-pct stop.

        When ``atr_stop_enabled`` and enough mark history exists, the stop is
        placed ``atr_stop_mult`` ATR-proxy widths from entry, but never closer
        than ``atr_stop_min_pct``. The ATR proxy is derived from per-tick mark
        returns, so without that floor the stop hugs the entry and any tick of
        noise stops the position out instantly. Otherwise the legacy
        :meth:`initial_stop_price` (fixed ``INITIAL_STOP_PCT``) is used.
        """

        if self.settings.atr_stop_enabled:
            atr_pct = self._tracker(coin).atr_pct
            if atr_pct is not None and atr_pct > 0:
                dist = atr_pct * self.settings.atr_stop_mult
                dist = max(dist, self.settings.atr_stop_min_pct)
                if side == SignalSide.LONG:
                    return entry_px * (Decimal("1") - dist)
                return entry_px * (Decimal("1") + dist)
        return self.initial_stop_price(entry_px, side)

    def _reversion_take_profit(
        self, position: ManagedPosition, mark_px: Decimal
    ) -> Decimal | None:
        """Return an exit price if the mean-reversion target has been reached.

        Only fires in profit (the hard/trailing stop handles adverse moves), so a
        long faded below the mean exits once price recovers to the mean, and a
        short faded above the mean exits once price falls back to it.
        """

        if not self.settings.reversion_tp_enabled:
            return None
        target = self._tracker(position.coin).reversion_target
        if target is None:
            return None
        if position.side == SignalSide.LONG:
            if mark_px >= target and mark_px > position.entry_px:
                return mark_px
        else:
            if mark_px <= target and mark_px < position.entry_px:
                return mark_px
        return None

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
        # A zero callback is "no trail", not "trail at the peak". Without this
        # the trigger equals the peak itself and every tick that fails to set a
        # new high closes the position.
        if callback <= 0:
            return None
        if position.side == SignalSide.LONG:
            trigger_px = position.peak_px * (Decimal("1") - callback)
            if mark_px <= trigger_px:
                return trigger_px
        else:
            trigger_px = position.peak_px * (Decimal("1") + callback)
            if mark_px >= trigger_px:
                return trigger_px
        return None

    def _within_min_hold(self, position: ManagedPosition, now: datetime) -> bool:
        """True if the position is younger than ``min_hold_seconds``.

        Suppresses hard-stop exits in the first moments after entry to avoid
        1-tick whipsaw stop-outs. The kill-switch panic path always overrides
        this (it runs before any per-position logic in ``on_price_update``).
        """
        min_hold = float(self.settings.min_hold_seconds)
        if min_hold <= 0:
            return False
        return (now - position.opened_at).total_seconds() < min_hold

    def _exceeded_max_hold(self, position: ManagedPosition, now: datetime) -> bool:
        """True once the position has been open longer than ``max_hold_seconds``.

        Evaluated after the price-based exits so a stop or trail that is already
        due still wins; the horizon exit only handles positions nothing else
        closed.
        """
        max_hold = float(self.settings.max_hold_seconds)
        if max_hold <= 0:
            return False
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return (now - opened_at).total_seconds() >= max_hold

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
            await self._journal.record_closed_trade(position, exit_px, pnl, roe_pct, reason)
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
            failed: list[str] = []
            for position in positions:
                exit_px = self.kill_switch.mark_prices.get(position.coin)
                if exit_px is None:
                    exit_px = position.peak_px or position.entry_px
                try:
                    await self._request_exit(position, "kill_switch", exit_px=exit_px)
                except Exception:  # noqa: BLE001 - one bad symbol must not strand the rest
                    # A panic close is the worst moment to abort a batch: the
                    # symbol that fails is often the one that is moving. Keep
                    # going and report, so the remaining positions still close.
                    failed.append(position.coin)
                    logger.exception("Panic close failed for %s", position.coin)
            if failed:
                logger.critical(
                    "Panic close incomplete; still open: %s", ", ".join(sorted(failed))
                )
        finally:
            self._panic_in_progress = False

    async def on_price_update(self, coin: str, mark_px: Decimal) -> None:
        if self.kill_switch.is_tripped:
            await self.panic_close_all()
            return

        normalized = coin.strip().upper()
        now = datetime.now(timezone.utc)

        # Detect a feed blackout (e.g. an upstream 502): a large gap since this
        # coin's previous mark. The first tick after a blackout is discontinuous,
        # so we re-anchor the trackers and skip all stop/exit decisions for that
        # one tick — otherwise a stop could fire at a gapped price. The next
        # fresh tick resumes normal management.
        last_ts = self._last_mark_ts.get(normalized)
        gap_limit = float(self.settings.mark_staleness_gap_seconds)
        stale_gap = (
            gap_limit > 0
            and last_ts is not None
            and (now - last_ts).total_seconds() > gap_limit
        )
        self._last_mark_ts[normalized] = now

        if stale_gap:
            self._tracker(normalized).resync(mark_px)
            logger.warning(
                "Stale mark gap for %s (>%.1fs); skipping stop checks for one tick",
                normalized,
                gap_limit,
            )
            return

        # Keep the volatility / reversion trackers warm even before a position
        # exists, so ATR-stop and reversion-TP are ready the moment one opens.
        self._tracker(normalized).update(mark_px)

        position = self._positions.get(normalized)
        if position is None:
            return

        previous_peak = position.peak_px
        self._update_peak(position, mark_px)
        if position.peak_px != previous_peak:
            await self._maybe_persist_position(position, now)

        await self.maybe_move_to_break_even(position, mark_px)

        if self._hit_hard_stop(position, mark_px):
            if self._within_min_hold(position, now):
                logger.debug(
                    "Hard stop for %s suppressed: within min-hold window", normalized
                )
            else:
                await self._request_exit(position, "stop_loss", exit_px=mark_px)
                return

        reversion_px = self._reversion_take_profit(position, mark_px)
        if reversion_px is not None:
            await self._request_exit(position, "reversion_tp", exit_px=reversion_px)
            return

        trail_trigger = await self.maybe_trail_stop(position, mark_px)
        if trail_trigger is not None:
            await self._request_exit(position, "trailing_stop", exit_px=mark_px)
            return

        if self._exceeded_max_hold(position, now):
            await self._request_exit(position, "time_stop", exit_px=mark_px)

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
