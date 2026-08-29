"""Measure how much of the dry-run edge survives realistic post-only fills.

The live bot places entries as ``Alo`` (post-only) limit orders at the signal's
mark price, and the dry-run path books every one of them as filled at that price.
A resting order does not work that way: it fills only when someone crosses to it,
which on a momentum entry means price came *back* through the order — the case
where the signal was wrong. The trades that quietly fail to fill are therefore
biased toward the profitable ones, and the dry-run number is an upper bound of
unknown tightness.

Three fill models are run over the same events and the same exit layer, so the
only thing that differs is how an entry is obtained:

  optimistic  fill at the signal mark, maker fee   -- what dry-run assumes
  realistic   post-only, must be touched, expires  -- what the exchange does
  taker       fill at the signal mark, taker fee   -- cross the spread instead

``optimistic`` vs ``realistic`` is the size of the assumption. ``taker`` is the
fallback if resting orders turn out not to fill: it pays 4.5bps instead of 1.5bps
on entry but fills every time, and the comparison says which is worth more.

Usage:
    python -m scripts.maker_fill_realism [recordings_dir] [--days N] [--from YYYY-MM-DD]

Prefer ``--from`` over ``--days``: the recording set straddles a 38-hour outage
around 2026-07-29, and a window that includes it is mostly gaps (see
scripts/sweep_time_stop.py for what that failure looks like).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal

from backtest.replay import recording_files, replay_file
from backtest.simulator import BacktestSimulator, SimConfig
from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.dvsla import DvslaParams, DvslaStrategy

_BPS = Decimal("10000")


class InstantFillSimulator(BacktestSimulator):
    """Fills entries at the signal mark, bypassing the resting-order queue.

    The fee rate still follows ``maker_entry_enabled``, so this reproduces the
    dry-run bookkeeping exactly when that flag is on: maker fee, guaranteed fill.
    """

    def _open_from_signal(self, signal) -> None:  # type: ignore[no-untyped-def]
        coin = signal.symbol.strip().upper()
        if coin in self._open or len(self._open) >= self._cfg.max_concurrent:
            self._result.skipped_signals += 1
            return
        if signal.entry_mark_price <= 0:
            self._result.skipped_signals += 1
            return
        confidence = max(Decimal("0"), min(Decimal("1"), signal.confidence))
        self._result.maker_orders_placed += 1
        self._result.maker_orders_filled += 1
        self._open_position(
            coin,
            signal.side,
            signal.entry_mark_price,
            signal.timestamp,
            confidence,
            signal.reason,
        )


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _tstat(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    sd = statistics.stdev(xs)
    return statistics.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="since", default=None, help="only files on/after YYYY-MM-DD")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="drop this day; repeatable. Use to test whether one session carries the result.",
    )
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))

    def cfg(maker: bool) -> SimConfig:
        return SimConfig(
            take_profit_pct=Decimal("0"),
            stop_loss_pct=settings.atr_stop_min_pct,
            time_stop_seconds=Decimal(str(settings.max_hold_seconds)),
            trailing_callback_pct=settings.trailing_callback_pct,
            break_even_trigger_pct=Decimal("0"),
            maker_entry_enabled=maker,
            maker_fee_bps=settings.maker_fee_bps,
            taker_fee_bps=settings.taker_fee_bps,
            maker_fill_timeout_seconds=Decimal(str(settings.max_hold_seconds)),
        )

    # maker=True -> maker fee on entry; maker=False -> taker fee on entry.
    models: list[tuple[str, BacktestSimulator]] = [
        ("optimistic", InstantFillSimulator(strategy, cfg(True))),
        ("realistic", BacktestSimulator(strategy, cfg(True))),
        ("taker", InstantFillSimulator(strategy, cfg(False))),
    ]

    files = recording_files(args.directory, since=args.since, exclude=args.exclude)
    if args.days:
        files = files[-args.days:]
    if not files:
        print("No recordings matched.")
        return 1

    print(f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})")
    print(
        f"Exit layer: time_stop={settings.max_hold_seconds}s  "
        f"trail={float(settings.trailing_callback_pct) * 100:.2f}%  "
        f"stop={float(settings.atr_stop_min_pct) * 100:.2f}%  "
        f"fees={float(settings.maker_fee_bps)}/{float(settings.taker_fee_bps)}bps\n"
    )

    async def run() -> None:
        n = 0
        t0 = time.time()
        for path in files:
            for event in replay_file(path):
                n += 1
                price, ts = _price_and_ts(event)
                if price is not None and event.coin is not None:
                    key = event.coin.strip().upper()
                    for _, sim in models:
                        sim._marks[key] = price
                        sim._check_exits(event.coin, price, ts)
                        sim._check_pending_fills(event.coin, price, ts)
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    for _, sim in models:
                        sim._open_from_signal(signal)
                if n % 20_000_000 == 0:
                    print(f"  ...{n:,} events ({n / (time.time() - t0):,.0f}/s)")

    asyncio.run(run())
    for _, sim in models:
        sim._close_remaining()

    print(f"{'model':>11} {'placed':>7} {'fill%':>6} {'n':>6} {'win%':>5} "
          f"{'net$':>10} {'mean bps':>9} {'t':>7}")
    print("-" * 72)
    summary: dict[str, list[float]] = {}
    for name, sim in models:
        r = sim._result
        nets = [float(t.net_pnl) for t in r.trades]
        bps = [float(t.net_pnl / t.notional * _BPS) for t in r.trades if t.notional]
        summary[name] = bps
        fill = (r.maker_orders_filled / r.maker_orders_placed * 100) if r.maker_orders_placed else 0.0
        if len(nets) < 2:
            print(f"{name:>11} {r.maker_orders_placed:7d} {fill:6.1f} {len(nets):6d}   (too few)")
            continue
        wins = sum(1 for x in nets if x > 0)
        print(f"{name:>11} {r.maker_orders_placed:7d} {fill:6.1f} {len(nets):6d} "
              f"{wins / len(nets) * 100:5.0f} {sum(nets):10.2f} "
              f"{statistics.mean(bps):9.2f} {_tstat(bps):+7.2f}")

    opt, real = summary.get("optimistic", []), summary.get("realistic", [])
    if opt and real:
        lost = statistics.mean(opt) - statistics.mean(real)
        print(f"\nAssumption cost: {lost:+.2f} bps/trade "
              f"({statistics.mean(opt):.2f} optimistic -> {statistics.mean(real):.2f} realistic)")

    # Per-day, so the edge can be lined up against market regime.
    print(f"\n{'day':>12} " + " ".join(f"{n[:9]:>18}" for n, _ in models))
    print("-" * (13 + 19 * len(models)))
    byday: dict[str, dict[date, list[float]]] = {n: defaultdict(list) for n, _ in models}
    for name, sim in models:
        for t in sim._result.trades:
            if t.notional:
                byday[name][t.exit_ts.date()].append(float(t.net_pnl / t.notional * _BPS))
    days = sorted({d for m in byday.values() for d in m})
    for d in days:
        cells = []
        for name, _ in models:
            v = byday[name][d]
            cells.append(f"{len(v):5d} {statistics.mean(v):+7.2f}bps" if v else f"{'-':>18}")
        print(f"{d!s:>12} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
