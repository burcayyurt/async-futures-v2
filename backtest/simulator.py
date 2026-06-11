"""Event-driven backtest simulator for the DVSLA strategy.

Drives a strategy engine through a replayed :class:`MarketEvent` stream exactly
as in live trading (same ``on_market_event`` interface), then models fills and
position lifecycle so the recorded signals translate into a realistic PnL curve.

Fill / exit model (self-contained, independent of the live execution layer so it
can be validated before PR5):

* **Entry** — maker limit assumed filled at the signal's ``entry_mark_price``;
  charged ``maker_fee_bps``. Position notional is scaled by signal confidence
  between ``min_notional`` and ``max_notional``.
* **Exit** — a bracket evaluated on every subsequent price print: a
  take-profit and a stop-loss as fixed percentages of entry, plus a time stop.
  Exits are taker fills charged ``taker_fee_bps`` plus ``slippage_bps``.

One position per symbol at a time; ``max_concurrent`` caps total open
positions. The simulator is deterministic and has no look-ahead: exits for a
signal can only trigger on prints at or after the entry event.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.signals import SignalSide, TradeSignal

_BPS = Decimal("10000")


@dataclass(slots=True)
class SimConfig:
    """Fill, sizing and exit parameters for the simulator."""

    maker_fee_bps: Decimal = Decimal("1.5")
    taker_fee_bps: Decimal = Decimal("4.5")
    slippage_bps: Decimal = Decimal("2.0")

    # Confidence-scaled position notional (quote currency).
    min_notional: Decimal = Decimal("100")
    max_notional: Decimal = Decimal("1000")

    # Bracket exit, as fractions of entry price.
    take_profit_pct: Decimal = Decimal("0.006")
    stop_loss_pct: Decimal = Decimal("0.004")
    # Time stop: close after this many seconds if neither bracket hit. 0 = off.
    time_stop_seconds: Decimal = Decimal("300")

    max_concurrent: int = 5


@dataclass(slots=True)
class OpenPosition:
    coin: str
    side: SignalSide
    entry_px: Decimal
    qty: Decimal
    notional: Decimal
    entry_ts: datetime
    stop_px: Decimal
    tp_px: Decimal
    risk: Decimal
    entry_fee: Decimal
    confidence: Decimal
    reason: str


@dataclass(slots=True)
class ClosedTrade:
    coin: str
    side: SignalSide
    entry_px: Decimal
    exit_px: Decimal
    qty: Decimal
    notional: Decimal
    entry_ts: datetime
    exit_ts: datetime
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    r_multiple: Decimal
    exit_reason: str
    confidence: Decimal


@dataclass
class BacktestResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    signals: int = 0
    skipped_signals: int = 0
    events_processed: int = 0


class BacktestSimulator:
    """Replays events through a strategy and books simulated trades."""

    def __init__(self, strategy, config: SimConfig | None = None) -> None:
        self._strategy = strategy
        self._cfg = config or SimConfig()
        self._open: dict[str, OpenPosition] = {}
        self._marks: dict[str, Decimal] = {}
        self._result = BacktestResult()

    # ------------------------------------------------------------------ run

    def run(self, events: Iterable[MarketEvent]) -> BacktestResult:
        """Synchronous entry point. Returns the populated result."""

        return asyncio.run(self.run_async(events))

    async def run_async(self, events: Iterable[MarketEvent]) -> BacktestResult:
        for event in events:
            self._result.events_processed += 1
            price, ts = self._price_and_ts(event)
            if price is not None:
                # Evaluate exits BEFORE the strategy sees the event so a signal
                # cannot close on its own entry print.
                self._check_exits(event.coin, price, ts)

            signal = await self._strategy.on_market_event(event)
            if signal is not None:
                self._result.signals += 1
                self._open_from_signal(signal)

        # Close anything still open at the last seen mark (mark-to-market).
        self._close_remaining()
        return self._result

    # -------------------------------------------------------------- helpers

    def _price_and_ts(self, event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
        if event.coin is None:
            return None, None
        coin = event.coin.strip().upper()
        if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
            self._marks[coin] = event.payload.px
            return event.payload.px, event.ts
        if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
            self._marks[coin] = event.payload.mark_px
            return event.payload.mark_px, event.ts
        return None, None

    def _open_from_signal(self, signal: TradeSignal) -> None:
        coin = signal.symbol.strip().upper()
        if coin in self._open or len(self._open) >= self._cfg.max_concurrent:
            self._result.skipped_signals += 1
            return
        entry_px = signal.entry_mark_price
        if entry_px <= 0:
            self._result.skipped_signals += 1
            return

        confidence = max(Decimal("0"), min(Decimal("1"), signal.confidence))
        notional = self._cfg.min_notional + (
            (self._cfg.max_notional - self._cfg.min_notional) * confidence
        )
        qty = notional / entry_px
        entry_fee = notional * self._cfg.maker_fee_bps / _BPS

        tp_pct = self._cfg.take_profit_pct
        sl_pct = self._cfg.stop_loss_pct
        if signal.side == SignalSide.LONG:
            tp_px = entry_px * (Decimal("1") + tp_pct)
            stop_px = entry_px * (Decimal("1") - sl_pct)
        else:
            tp_px = entry_px * (Decimal("1") - tp_pct)
            stop_px = entry_px * (Decimal("1") + sl_pct)

        self._open[coin] = OpenPosition(
            coin=coin,
            side=signal.side,
            entry_px=entry_px,
            qty=qty,
            notional=notional,
            entry_ts=signal.timestamp,
            stop_px=stop_px,
            tp_px=tp_px,
            risk=notional * sl_pct,
            entry_fee=entry_fee,
            confidence=confidence,
            reason=signal.reason,
        )

    def _check_exits(self, coin: str | None, price: Decimal, ts: datetime | None) -> None:
        if coin is None:
            return
        key = coin.strip().upper()
        pos = self._open.get(key)
        if pos is None or ts is None:
            return

        exit_px: Decimal | None = None
        reason = ""
        if pos.side == SignalSide.LONG:
            if price <= pos.stop_px:
                exit_px, reason = pos.stop_px, "stop"
            elif price >= pos.tp_px:
                exit_px, reason = pos.tp_px, "take_profit"
        else:
            if price >= pos.stop_px:
                exit_px, reason = pos.stop_px, "stop"
            elif price <= pos.tp_px:
                exit_px, reason = pos.tp_px, "take_profit"

        if exit_px is None and self._cfg.time_stop_seconds > 0:
            elapsed = Decimal(str((ts - pos.entry_ts).total_seconds()))
            if elapsed >= self._cfg.time_stop_seconds:
                exit_px, reason = price, "time_stop"

        if exit_px is not None:
            self._book_exit(pos, exit_px, ts, reason)
            del self._open[key]

    def _close_remaining(self) -> None:
        for key, pos in list(self._open.items()):
            mark = self._marks.get(key, pos.entry_px)
            self._book_exit(pos, mark, pos.entry_ts, "eod_mark")
            del self._open[key]

    def _book_exit(
        self,
        pos: OpenPosition,
        exit_px: Decimal,
        ts: datetime,
        reason: str,
    ) -> None:
        # Apply taker slippage against the position on exit.
        slip = exit_px * self._cfg.slippage_bps / _BPS
        if pos.side == SignalSide.LONG:
            fill_px = exit_px - slip
            gross = (fill_px - pos.entry_px) * pos.qty
        else:
            fill_px = exit_px + slip
            gross = (pos.entry_px - fill_px) * pos.qty

        exit_notional = fill_px * pos.qty
        exit_fee = exit_notional * self._cfg.taker_fee_bps / _BPS
        fees = pos.entry_fee + exit_fee
        net = gross - fees
        r_multiple = net / pos.risk if pos.risk > 0 else Decimal("0")

        self._result.trades.append(
            ClosedTrade(
                coin=pos.coin,
                side=pos.side,
                entry_px=pos.entry_px,
                exit_px=fill_px,
                qty=pos.qty,
                notional=pos.notional,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                gross_pnl=gross,
                fees=fees,
                net_pnl=net,
                r_multiple=r_multiple,
                exit_reason=reason,
                confidence=pos.confidence,
            )
        )
