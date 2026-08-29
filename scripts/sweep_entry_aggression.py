"""Ask what the entry quote costs, and whether quoting closer recovers it.

At the first tick after a signal the market has already moved about 14 bps in
the signal's own direction, and it stays there — the drift is flat from 15 ms
out to 500 ms (``scripts/latency_cost.py``). The entry therefore does not sit
behind the market because of network delay; it sits behind because of where it
is quoted. The bot rests a post-only order at the signal's mark, and the market
has left that price. The order either never fills, or fills precisely when the
move failed and price came back — the adverse-selection channel named in every
earlier study here.

That 14 bps is larger than any per-trade edge yet measured on this strategy,
which makes the quote the largest single lever available. This sweep maps it.

For a signal at mark ``P0``, aggressiveness ``k`` moves the limit ``k`` bps in
the signal's direction: a long bids above the signal mark rather than at it.
The order is then subject to the rule that makes this non-trivial — a post-only
order that would cross the book on arrival is **rejected by the exchange**, not
filled. Without that, the sweep could quote arbitrarily aggressively and report
free money. Two policies are swept:

  alo    a crossing order is rejected and the trade is lost
  chase  a crossing order falls back to IOC and pays the taker rate

``alo +0`` is what runs in production today, so it anchors the grid. Full taker
is the far end, already measured as the *worst* option once the two cascade
sessions are excluded — so the interesting region is the middle, which nothing
has looked at yet.

Arrival is modelled as the first recorded tick after the signal. The recordings
carry this machine's own view, so that tick is already latency-delayed; the
model cannot resolve anything finer and does not pretend to.

Usage:
    python -m scripts.sweep_entry_aggression [dir] [--from YYYY-MM-DD]
                                             [--exclude YYYY-MM-DD]... [--bps 0,2,5]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from backtest.replay import recording_files, replay_file
from backtest.simulator import BacktestSimulator, PendingMakerOrder, SimConfig
from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.dvsla import DvslaParams, DvslaStrategy
from src.strategy.signals import SignalSide, TradeSignal

_BPS = Decimal("10000")


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


class AggressiveEntrySimulator(BacktestSimulator):
    """Post-only entries quoted ``offset_bps`` into the signal's direction.

    The base simulator rests every order at the signal mark, which can never
    cross and so can never be rejected. Once the quote moves toward the market
    that stops being true, so this adds the arrival check a real exchange
    applies to a post-only order.
    """

    def __init__(self, cfg: SimConfig, *, offset_bps: Decimal, on_cross: str) -> None:
        super().__init__(None, cfg)
        self._offset_bps = offset_bps
        self._on_cross = on_cross  # "alo" (reject) or "chase" (IOC at market)
        self._unarrived: set[str] = set()
        self.rejected = 0
        self.chased = 0

    def _open_from_signal(self, signal: TradeSignal) -> None:
        coin = signal.symbol.strip().upper()
        if (
            coin in self._open
            or coin in self._pending
            or len(self._open) + len(self._pending) >= self._cfg.max_concurrent
        ):
            self._result.skipped_signals += 1
            return
        if signal.entry_mark_price <= 0:
            self._result.skipped_signals += 1
            return

        sign = Decimal("1") if signal.side == SignalSide.LONG else Decimal("-1")
        limit_px = signal.entry_mark_price * (Decimal("1") + sign * self._offset_bps / _BPS)

        self._pending[coin] = PendingMakerOrder(
            coin=coin,
            side=signal.side,
            limit_px=limit_px,
            placed_ts=signal.timestamp,
            confidence=max(Decimal("0"), min(Decimal("1"), signal.confidence)),
            reason=signal.reason,
        )
        self._unarrived.add(coin)
        self._result.maker_orders_placed += 1

    def _open_taker(self, pending: PendingMakerOrder, price: Decimal, ts: datetime) -> None:
        # _open_position reads the fee rate off the config flag, so the flag is
        # what has to move for this one fill to be charged as a taker.
        was_maker = self._cfg.maker_entry_enabled
        self._cfg.maker_entry_enabled = False
        try:
            self._open_position(
                pending.coin, pending.side, price, ts, pending.confidence, pending.reason
            )
        finally:
            self._cfg.maker_entry_enabled = was_maker

    def _check_pending_fills(
        self, coin: str | None, price: Decimal, ts: datetime | None
    ) -> None:
        if coin is None or ts is None:
            return
        key = coin.strip().upper()
        pending = self._pending.get(key)
        if pending is None:
            return

        if key in self._unarrived:
            self._unarrived.discard(key)
            # Strictly beyond, not "at or beyond": the recorded price is a trade
            # print or a mark, and a bid resting exactly there does not cross the
            # ask. The comparison is still crude — without book data the last
            # price stands in for the far touch, so a buy is judged against a
            # print that may have happened at the bid. That over-reports
            # crossing, which biases this study against aggressive quoting; the
            # conservative direction for a decision of this kind.
            crosses = (
                pending.limit_px > price
                if pending.side == SignalSide.LONG
                else pending.limit_px < price
            )
            if crosses:
                del self._pending[key]
                if self._on_cross == "chase":
                    self.chased += 1
                    self._result.maker_orders_filled += 1
                    self._open_taker(pending, price, ts)
                else:
                    self.rejected += 1
                    self._result.skipped_signals += 1
                return
            # Resting. It cannot also fill on this tick: for a long, "not
            # crossing" means price is still above the bid.
            return

        super()._check_pending_fills(coin, price, ts)


class TakerSimulator(BacktestSimulator):
    """Every entry crosses at the first tick after the signal, paying taker."""

    def __init__(self, cfg: SimConfig) -> None:
        cfg.maker_entry_enabled = False
        super().__init__(None, cfg)
        self._await_fill: dict[str, PendingMakerOrder] = {}

    def _open_from_signal(self, signal: TradeSignal) -> None:
        coin = signal.symbol.strip().upper()
        if coin in self._open or coin in self._await_fill:
            self._result.skipped_signals += 1
            return
        if len(self._open) + len(self._await_fill) >= self._cfg.max_concurrent:
            self._result.skipped_signals += 1
            return
        self._await_fill[coin] = PendingMakerOrder(
            coin=coin,
            side=signal.side,
            limit_px=signal.entry_mark_price,
            placed_ts=signal.timestamp,
            confidence=max(Decimal("0"), min(Decimal("1"), signal.confidence)),
            reason=signal.reason,
        )

    def _check_pending_fills(
        self, coin: str | None, price: Decimal, ts: datetime | None
    ) -> None:
        if coin is None or ts is None:
            return
        key = coin.strip().upper()
        order = self._await_fill.pop(key, None)
        if order is None:
            return
        # Crossing at the arrival tick's price is the honest taker assumption:
        # the same information a resting order would have had at that moment.
        self._open_position(key, order.side, price, ts, order.confidence, order.reason)


def _tstat(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    sd = statistics.stdev(xs)
    return statistics.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument("--to", dest="until", default=None)
    ap.add_argument("--exclude", action="append", default=[], metavar="YYYY-MM-DD")
    ap.add_argument("--bps", default="0,2,5,10,15,20")
    args = ap.parse_args()
    offsets = [Decimal(b.strip()) for b in args.bps.split(",") if b.strip()]

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))
    floor = settings.dvsla_min_confidence

    def cfg() -> SimConfig:
        return SimConfig(
            take_profit_pct=Decimal("0"),
            stop_loss_pct=settings.atr_stop_min_pct,
            time_stop_seconds=Decimal(str(settings.max_hold_seconds)),
            trailing_callback_pct=settings.trailing_callback_pct,
            break_even_trigger_pct=Decimal("0"),
            maker_entry_enabled=True,
            maker_fee_bps=settings.maker_fee_bps,
            taker_fee_bps=settings.taker_fee_bps,
            maker_fill_timeout_seconds=Decimal(str(settings.max_hold_seconds)),
        )

    variants: list[tuple[str, BacktestSimulator]] = []
    for policy in ("alo", "chase"):
        for k in offsets:
            if policy == "chase" and k == 0:
                continue  # k=0 never crosses, so chase would duplicate alo
            variants.append(
                (
                    f"{policy} +{k}",
                    AggressiveEntrySimulator(cfg(), offset_bps=k, on_cross=policy),
                )
            )
    variants.append(("taker", TakerSimulator(cfg())))

    files = recording_files(
        args.directory, since=args.since, until=args.until, exclude=args.exclude
    )
    if not files:
        print("No recordings matched.")
        return 1

    print(
        f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})"
        + (f"  excluded: {','.join(sorted(args.exclude))}" if args.exclude else "")
    )
    print(
        f"Confidence floor {float(floor):.2f}, fees {float(settings.maker_fee_bps)}/"
        f"{float(settings.taker_fee_bps)}bps, time stop {settings.max_hold_seconds}s\n"
    )

    async def run() -> None:
        for path in files:
            for event in replay_file(path):
                price, ts = _price_and_ts(event)
                coin = event.coin.strip().upper() if event.coin else None
                for _, sim in variants:
                    if price is not None and coin is not None:
                        sim._marks[coin] = price
                        sim._check_exits(event.coin, price, ts)
                        sim._check_pending_fills(event.coin, price, ts)
                signal = await strategy.on_market_event(event)
                if signal is not None and signal.confidence >= floor:
                    for _, sim in variants:
                        sim._open_from_signal(signal)

    asyncio.run(run())
    for _, sim in variants:
        sim._close_remaining()

    header = (
        f"{'variant':>10} {'n':>5} {'fill%':>6} {'red':>5} {'win%':>5} "
        f"{'net$':>9} {'bps':>7} {'t':>7} {'best day%':>10} {'ex-top5':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, sim in variants:
        r = sim._result
        trades = [t for t in r.trades if t.notional]
        if len(trades) < 2:
            print(f"{name:>10} {len(trades):5d}   (too few)")
            continue
        bps = [float(t.net_pnl / t.notional * _BPS) for t in trades]
        nets = [float(t.net_pnl) for t in trades]
        placed = r.maker_orders_placed or len(trades)
        fill = len(trades) / placed * 100 if placed else float("nan")
        rejected = getattr(sim, "rejected", 0)

        byday: dict[object, list[float]] = defaultdict(list)
        for t, b in zip(trades, bps):
            byday[t.exit_ts.date()].append(b)
        contrib = sorted(((sum(v), d) for d, v in byday.items()), reverse=True)
        total = sum(bps)
        best = (contrib[0][0] / total * 100) if total else float("nan")
        top5 = {d for _, d in contrib[:5]}
        rest = [b for t, b in zip(trades, bps) if t.exit_ts.date() not in top5]
        ex5 = statistics.mean(rest) if rest else float("nan")

        print(
            f"{name:>10} {len(trades):5d} {fill:6.1f} {rejected:5d} "
            f"{sum(1 for x in nets if x > 0) / len(nets) * 100:5.0f} {sum(nets):9.2f} "
            f"{statistics.mean(bps):7.2f} {_tstat(bps):+7.2f} {best:9.0f}% {ex5:+8.2f}"
        )

    print("\nred     = post-only karsiya gectigi icin borsanin reddettigi sinyal sayisi")
    print("alo +0  = bugun canlida calisan davranis")
    print("ex-top5 = en iyi bes gun haric ortalama bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
