"""Efficient TP/SL/time-stop sweep over recorded DVSLA events.

The DVSLA signal stream depends only on the *entry* params (``DvslaParams``), not
on the exit bracket (``SimConfig.take_profit_pct`` / ``stop_loss_pct`` /
``time_stop_seconds``). So instead of replaying the whole (multi-GB) recording
once per grid point, we replay it **once**, run the strategy **once**, and fan
each produced signal out to N independent bracket simulators that share the same
single event/price pass. Cost ~= 1 strategy pass + N cheap bracket checks.

Entry params are read from the live ``.env`` (so the sweep optimises the exit
layer for the *current* signal generator). Override the grid below as needed.

Usage:
    python scripts/sweep_tp_sl.py [recordings_dir] [--days N] [--max-events N]
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backtest.metrics import compute_metrics
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
from src.strategy.signals import SignalSide


def _flip(signal):
    other = SignalSide.SHORT if signal.side == SignalSide.LONG else SignalSide.LONG
    return signal.model_copy(update={"side": other})


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _grid() -> dict[str, list[Decimal]]:
    return {
        "take_profit_pct": [Decimal(x) for x in ("0.004", "0.006", "0.008", "0.010", "0.015")],
        "stop_loss_pct": [Decimal(x) for x in ("0.003", "0.004", "0.006", "0.008")],
        "time_stop_seconds": [Decimal(x) for x in ("180", "300", "600")],
    }


def _iter_files(directory: str, days: int | None, first: int | None) -> list[Path]:
    files = recording_files(directory)
    if first is not None and first > 0:
        files = files[:first]
    elif days is not None and days > 0:
        files = files[-days:]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None, help="use only the last N day files")
    ap.add_argument("--first", type=int, default=None, help="use only the first N day files")
    ap.add_argument("--max-events", type=int, default=None, help="cap events (smoke test)")
    ap.add_argument("--invert", action="store_true", help="trade WITH the cascade (flip signal side)")
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    strategy = DvslaStrategy(params)

    grid = _grid()
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    configs = [SimConfig(**c) for c in combos]
    sims = [BacktestSimulator(strategy, cfg) for cfg in configs]

    files = _iter_files(args.directory, args.days, args.first)
    print(f"Files: {[f.name for f in files]}")
    print(f"Grid points: {len(configs)}  (TP x SL x time_stop)")

    import asyncio

    async def run() -> int:
        n = 0
        t0 = time.time()
        for path in files:
            for event in replay_file(path):
                n += 1
                if args.max_events is not None and n > args.max_events:
                    return n
                price, ts = _price_and_ts(event)
                if price is not None and event.coin is not None:
                    key = event.coin.strip().upper()
                    for sim in sims:
                        sim._marks[key] = price
                        sim._check_exits(event.coin, price, ts)
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    if args.invert:
                        signal = _flip(signal)
                    for sim in sims:
                        sim._open_from_signal(signal)
                if n % 2_000_000 == 0:
                    print(f"  ...{n:,} events ({n / (time.time() - t0):,.0f}/s)")
        return n

    n = asyncio.run(run())
    for sim in sims:
        sim._close_remaining()
    print(f"Replayed {n:,} events\n")

    rows = []
    for combo, sim in zip(combos, sims):
        m = compute_metrics(sim._result.trades)
        rows.append((combo, m))

    # Rank: enough trades first, then profit_factor, then expectancy.
    def key(row):
        combo, m = row
        pf = m.profit_factor if m.profit_factor != float("inf") else 1e9
        return (1 if m.trades >= 10 else 0, pf, float(m.expectancy))

    rows.sort(key=key, reverse=True)

    print(f"{'TP%':>6} {'SL%':>6} {'tstop':>6} {'n':>4} {'win%':>5} "
          f"{'PF':>6} {'exp$':>8} {'netPnL$':>9} {'avgR':>6} {'maxDD$':>8}")
    print("-" * 78)
    for combo, m in rows[:25]:
        print(
            f"{float(combo['take_profit_pct']) * 100:6.2f} "
            f"{float(combo['stop_loss_pct']) * 100:6.2f} "
            f"{int(combo['time_stop_seconds']):6d} "
            f"{m.trades:4d} {m.win_rate * 100:5.0f} "
            f"{m.profit_factor:6.2f} {float(m.expectancy):8.3f} "
            f"{float(m.net_pnl):9.2f} {m.avg_r:6.2f} {float(m.max_drawdown):8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
