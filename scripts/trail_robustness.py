"""Stress-test a handful of trailing configs for tail dependence.

A sweep reports the mean. A mean built from three lucky trades is not an edge, so
before acting on a grid point this replays the same recordings and reports the
*distribution*: t-statistic of per-trade net PnL, how much of the profit the top
handful of trades contributed, and what survives with those trades removed.

Usage:
    python -m scripts.trail_robustness [recordings_dir] [--days N]
"""

from __future__ import annotations

import argparse
import asyncio
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

# (trail, break_even) as fractions — the sweep's survivors plus the live baseline.
CONFIGS = [
    ("0.0035", "0.0065"),  # live today
    ("0.015", "0"),
    ("0.020", "0"),
    ("0.030", "0"),
    ("0.040", "0"),
]


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _report(label: str, nets: list[float]) -> None:
    n = len(nets)
    if n < 2:
        print(f"{label}: too few trades ({n})")
        return
    total = sum(nets)
    mean = statistics.mean(nets)
    sd = statistics.stdev(nets)
    t = mean / (sd / n**0.5) if sd > 0 else float("nan")
    ranked = sorted(nets, reverse=True)
    top1, top3, top5 = ranked[0], sum(ranked[:3]), sum(ranked[:5])
    ex5 = sum(ranked[5:])
    wins = sum(1 for x in nets if x > 0)

    print(f"\n=== {label} ===")
    print(f"  trades      : {n}   win rate: {wins / n * 100:.0f}%")
    print(f"  net total   : {total:+.2f}   mean/trade: {mean:+.4f}   sd: {sd:.4f}")
    print(f"  t-stat      : {t:+.2f}   {'SIGNIFICANT' if abs(t) >= 2 else 'not significant'}")
    print(f"  best trade  : {top1:+.2f}  ({top1 / total * 100:.0f}% of net)" if total else "")
    print(f"  top 3       : {top3:+.2f}  ({top3 / total * 100:.0f}% of net)" if total else "")
    print(f"  net w/o top5: {ex5:+.2f}   <- edge without the tail")
    print(f"  median      : {statistics.median(nets):+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))

    sims = [
        BacktestSimulator(
            strategy,
            SimConfig(
                take_profit_pct=Decimal("0"),
                stop_loss_pct=settings.atr_stop_min_pct,
                time_stop_seconds=Decimal("0"),
                maker_fee_bps=settings.maker_fee_bps,
                taker_fee_bps=settings.taker_fee_bps,
                trailing_callback_pct=Decimal(t),
                break_even_trigger_pct=Decimal(b),
            ),
        )
        for t, b in CONFIGS
    ]

    files = recording_files(args.directory)
    if args.days:
        files = files[-args.days:]
    print(f"Files: {len(files)}   Configs: {len(CONFIGS)}")

    async def run() -> int:
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
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    for sim in sims:
                        sim._open_from_signal(signal)
                if n % 10_000_000 == 0:
                    print(f"  ...{n:,} ({n / (time.time() - t0):,.0f}/s)")
        return n

    asyncio.run(run())
    for sim in sims:
        sim._close_remaining()

    for (t, b), sim in zip(CONFIGS, sims):
        label = f"trail {float(t) * 100:.2f}%  BE {float(b) * 100:.2f}%"
        _report(label, [float(tr.net_pnl) for tr in sim._result.trades])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
