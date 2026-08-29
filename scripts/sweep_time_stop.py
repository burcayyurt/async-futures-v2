"""Test the horizon-exit hypothesis against the trailing exit.

`scripts/signal_forward_returns.py` shows the DVSLA signal carries a real,
fee-clearing edge (+0.11% at 30-60s, t > 10) while the live bot still loses. That
points at the exit, not the entry: a trail seeded at the entry price truncates
the upside the signal is predicting.

This sweeps a fixed-horizon exit against the current trailing exit, with and
without realistic post-only fills, so the comparison is like-for-like and net of
fees.

Usage:
    python -m scripts.sweep_time_stop [recordings_dir] [--days N]

Careful with small ``--days``: the newest recordings straddle a 38-hour outage
(2026-07-29 is ~6.6MB against a normal ~600MB), so a short window is mostly
gaps. Consecutive ticks for a coin can then be hours apart, every position hits
its time stop on the first tick after entry, and the output looks like broken
exit logic. Prefer the full set.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import statistics
import time
from datetime import datetime
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

TIME_STOPS = ["30", "60", "120", "300", "600"]
TRAILS = ["0", "0.0035"]        # off vs the live setting
MAKER = [False, True]           # instant fill vs realistic post-only


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument("--to", dest="until", default=None)
    ap.add_argument("--exclude", action="append", default=[], metavar="YYYY-MM-DD")
    ap.add_argument(
        "--time-stops",
        default=",".join(TIME_STOPS),
        help="hold horizons in seconds. Worth pushing well past a minute: the "
             "signal's forward return keeps growing out to four hours once the "
             "cascade sessions are excluded, and a longer hold is the only way "
             "to stop competing on entry speed.",
    )
    ap.add_argument("--trails", default=",".join(TRAILS))
    ap.add_argument(
        "--stops",
        default=None,
        help="hard-stop distances as fractions, e.g. 0.012,0.025,0. A stop "
             "calibrated for a one-minute hold is not a fair companion to a "
             "two-hour one: over the longer window ordinary noise reaches it, "
             "and each trigger costs the full stop distance.",
    )
    ap.add_argument(
        "--maker",
        default="true",
        help="true (realistic post-only), false (instant fill), or both",
    )
    args = ap.parse_args()

    time_stops = [t.strip() for t in args.time_stops.split(",") if t.strip()]
    trails = [t.strip() for t in args.trails.split(",") if t.strip()]
    maker = {"true": [True], "false": [False], "both": [False, True]}[args.maker.lower()]

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))

    stops = (
        [Decimal(x.strip()) for x in args.stops.split(",") if x.strip()]
        if args.stops
        else [settings.atr_stop_min_pct]
    )
    combos = list(itertools.product(time_stops, trails, maker, stops))
    sims = [
        BacktestSimulator(
            strategy,
            SimConfig(
                take_profit_pct=Decimal("0"),
                stop_loss_pct=sl,
                time_stop_seconds=Decimal(ts),
                trailing_callback_pct=Decimal(tr),
                break_even_trigger_pct=Decimal("0"),
                maker_entry_enabled=mk,
                maker_fee_bps=settings.maker_fee_bps,
                taker_fee_bps=settings.taker_fee_bps,
            ),
        )
        for ts, tr, mk, sl in combos
    ]

    files = recording_files(
        args.directory, since=args.since, until=args.until, exclude=args.exclude
    )
    if args.days:
        files = files[-args.days:]
    print(f"Files: {len(files)}   Grid: {len(combos)} (time-stop x trail x maker x stop)"
          + (f"   excluded: {','.join(sorted(args.exclude))}" if args.exclude else ""))
    print(f"Fixed: TP=off  stop={float(settings.atr_stop_min_pct) * 100:.2f}%  "
          f"fees={float(settings.maker_fee_bps)}/{float(settings.taker_fee_bps)}bps\n")

    async def run() -> None:
        n = 0
        t0 = time.time()
        for path in files:
            for event in replay_file(path):
                n += 1
                price, ts = _price_and_ts(event)
                if price is not None and event.coin is not None:
                    key = event.coin.strip().upper()
                    for sim in sims:
                        sim._marks[key] = price
                        sim._check_exits(event.coin, price, ts)
                        sim._check_pending_fills(event.coin, price, ts)
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    for sim in sims:
                        sim._open_from_signal(signal)
                if n % 20_000_000 == 0:
                    print(f"  ...{n:,} ({n / (time.time() - t0):,.0f}/s)")

    asyncio.run(run())
    for sim in sims:
        sim._close_remaining()

    # bps alongside dollars because every other study here reports bps, and
    # skipped because a long hold occupies one of five slots for its whole
    # length — a horizon can look good per trade and still take far fewer.
    print(f"{'tstop':>6} {'stop%':>6} {'maker':>6} {'fill%':>6} {'n':>5} {'skip':>6} "
          f"{'win%':>5} {'net$':>9} {'bps':>7} {'t':>7}  exit reasons")
    print("-" * 108)
    for (ts, tr, mk, sl), sim in zip(combos, sims):
        nets = [float(t.net_pnl) for t in sim._result.trades]
        r = sim._result
        placed = r.maker_orders_placed
        fill_rate = (r.maker_orders_filled / placed * 100) if placed else 100.0
        if len(nets) < 2:
            print(f"{ts:>6} {float(sl) * 100:6.2f} {str(mk):>6} {fill_rate:6.0f} {len(nets):5d}"
                  f"   (too few trades)")
            continue
        bps = [float(t.net_pnl / t.notional * Decimal("10000"))
               for t in r.trades if t.notional]
        mean = statistics.mean(nets)
        sd = statistics.stdev(nets)
        tstat = mean / (sd / len(nets) ** 0.5) if sd > 0 else float("nan")
        wins = sum(1 for x in nets if x > 0)
        reasons: dict[str, int] = {}
        for tr_ in sim._result.trades:
            reasons[tr_.exit_reason] = reasons.get(tr_.exit_reason, 0) + 1
        reason_str = " ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        mean_bps = statistics.mean(bps) if bps else float("nan")
        print(f"{ts:>6} {float(sl) * 100:6.2f} {str(mk):>6} {fill_rate:6.0f} {len(nets):5d} "
              f"{r.skipped_signals:6d} {wins / len(nets) * 100:5.0f} {sum(nets):9.2f} "
              f"{mean_bps:7.2f} {tstat:+7.2f}  {reason_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
