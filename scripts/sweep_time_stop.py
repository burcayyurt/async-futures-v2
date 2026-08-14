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
from pathlib import Path

from backtest.replay import replay_file
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
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))

    combos = list(itertools.product(TIME_STOPS, TRAILS, MAKER))
    sims = [
        BacktestSimulator(
            strategy,
            SimConfig(
                take_profit_pct=Decimal("0"),
                stop_loss_pct=settings.atr_stop_min_pct,
                time_stop_seconds=Decimal(ts),
                trailing_callback_pct=Decimal(tr),
                break_even_trigger_pct=Decimal("0"),
                maker_entry_enabled=mk,
                maker_fee_bps=settings.maker_fee_bps,
                taker_fee_bps=settings.taker_fee_bps,
            ),
        )
        for ts, tr, mk in combos
    ]

    files = sorted(Path(args.directory).glob("events-*.jsonl"))
    if args.days:
        files = files[-args.days:]
    print(f"Files: {len(files)}   Grid: {len(combos)} (time-stop x trail x maker-fill)")
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

    print(f"{'tstop':>6} {'trail%':>7} {'maker':>6} {'fill%':>6} {'n':>5} {'win%':>5} "
          f"{'net$':>9} {'mean$':>8} {'t':>7}  exit reasons")
    print("-" * 96)
    for (ts, tr, mk), sim in zip(combos, sims):
        nets = [float(t.net_pnl) for t in sim._result.trades]
        r = sim._result
        placed = r.maker_orders_placed
        fill_rate = (r.maker_orders_filled / placed * 100) if placed else 100.0
        if len(nets) < 2:
            print(f"{ts:>6} {float(tr) * 100:7.2f} {str(mk):>6} {fill_rate:6.0f} {len(nets):5d}"
                  f"   (too few trades)")
            continue
        mean = statistics.mean(nets)
        sd = statistics.stdev(nets)
        tstat = mean / (sd / len(nets) ** 0.5) if sd > 0 else float("nan")
        wins = sum(1 for x in nets if x > 0)
        reasons: dict[str, int] = {}
        for tr_ in sim._result.trades:
            reasons[tr_.exit_reason] = reasons.get(tr_.exit_reason, 0) + 1
        reason_str = " ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        print(f"{ts:>6} {float(tr) * 100:7.2f} {str(mk):>6} {fill_rate:6.0f} {len(nets):5d} "
              f"{wins / len(nets) * 100:5.0f} {sum(nets):9.2f} {mean:8.4f} {tstat:+7.2f}  {reason_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
