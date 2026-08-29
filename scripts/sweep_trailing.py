"""Sweep the trailing-stop callback (and break-even trigger) over recorded events.

Same single-pass trick as ``sweep_tp_sl.py``: the DVSLA signal stream depends
only on the *entry* params, so the recording is replayed once, the strategy runs
once, and every produced signal is fanned out to N exit simulators.

The grid models the **live** exit layer rather than the legacy TP/SL bracket:
fixed take-profit off, stop at the ATR floor (``ATR_STOP_MIN_PCT``), and the
trail/break-even doing the work — matching ``src.execution.position_manager``.

Unit note: ``TRAILING_CALLBACK_PCT`` is a *fraction* in the live config while
``BREAK_EVEN_TRIGGER_PCT`` is a *percent*. SimConfig takes both as fractions, so
the live break-even of ``0.65`` maps to ``0.0065`` here.

Entry params (including ``DVSLA_INVERT``) come from the live ``.env``, so do NOT
pass an extra invert flag — the strategy already applies it internally.

Usage:
    python -m scripts.sweep_trailing [recordings_dir] [--days N] [--max-events N]

Careful with small ``--days``: the newest recordings straddle a 38-hour outage,
so a short window is mostly gaps and produces misleading exit statistics. Prefer
the full set.
"""

from __future__ import annotations

import argparse
import asyncio
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

# Trail widths to test, as fractions. 0.0035 is what the bot runs today.
TRAIL_GRID = ["0.0035", "0.005", "0.0075", "0.010", "0.015", "0.020", "0.030"]
# Break-even trigger as a fraction: 0 = off, 0.0065 = the live 0.65%.
BE_GRID = ["0", "0.0065"]


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _iter_files(directory: str, days: int | None, first: int | None) -> list[Path]:
    files = recording_files(directory)
    if first is not None and first > 0:
        return files[:first]
    if days is not None and days > 0:
        return files[-days:]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None, help="use only the last N day files")
    ap.add_argument("--first", type=int, default=None, help="use only the first N day files")
    ap.add_argument("--max-events", type=int, default=None, help="cap events (smoke test)")
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    strategy = DvslaStrategy(params)

    stop_pct = settings.atr_stop_min_pct
    combos = [
        {"trailing_callback_pct": Decimal(t), "break_even_trigger_pct": Decimal(b)}
        for t, b in itertools.product(TRAIL_GRID, BE_GRID)
    ]
    configs = [
        SimConfig(
            take_profit_pct=Decimal("0"),
            stop_loss_pct=stop_pct,
            time_stop_seconds=Decimal("0"),
            maker_fee_bps=settings.maker_fee_bps,
            taker_fee_bps=settings.taker_fee_bps,
            **c,
        )
        for c in combos
    ]
    sims = [BacktestSimulator(strategy, cfg) for cfg in configs]

    files = _iter_files(args.directory, args.days, args.first)
    print(f"Files      : {len(files)} ({files[0].name} .. {files[-1].name})" if files else "no files")
    print(f"Engine     : dvsla invert={params.invert}")
    print(f"Fixed      : TP=off  stop={float(stop_pct) * 100:.2f}%  "
          f"fees={float(settings.maker_fee_bps)}/{float(settings.taker_fee_bps)}bps")
    print(f"Grid       : {len(configs)} points (trail x break-even)\n")

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
                    for sim in sims:
                        sim._open_from_signal(signal)
                if n % 2_000_000 == 0:
                    print(f"  ...{n:,} events ({n / (time.time() - t0):,.0f}/s)")
        return n

    n = asyncio.run(run())
    for sim in sims:
        sim._close_remaining()
    print(f"\nReplayed {n:,} events\n")

    rows = [(c, compute_metrics(s._result.trades)) for c, s in zip(combos, sims)]

    print(f"{'trail%':>7} {'BE%':>5} {'n':>5} {'win%':>5} {'PF':>6} "
          f"{'exp$':>8} {'netPnL$':>9} {'avgR':>6} {'maxDD$':>8}")
    print("-" * 70)
    for combo, m in rows:
        be = float(combo["break_even_trigger_pct"]) * 100
        pf = m.profit_factor if m.profit_factor != float("inf") else float("nan")
        print(
            f"{float(combo['trailing_callback_pct']) * 100:7.2f} "
            f"{be:5.2f} {m.trades:5d} {m.win_rate * 100:5.0f} {pf:6.2f} "
            f"{float(m.expectancy):8.4f} {float(m.net_pnl):9.2f} "
            f"{m.avg_r:6.2f} {float(m.max_drawdown):8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
