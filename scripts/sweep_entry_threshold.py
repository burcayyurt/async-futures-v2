"""Ask whether raising the entry bar cuts the bleed without cutting the payday.

Over the full recording set the strategy earns +2.85 bps/trade with realistic
post-only fills, but removing the best five sessions of 52 turns that into
-0.22 bps. The return lives in rare cascades; the ordinary days leak. If a
stricter entry filter removes mostly leak, the system improves. If it also
removes the rare days, the entry thesis is what needs revisiting, not its
thresholds — which is the useful thing to learn either way.

Two bars are swept together because they gate different things:

  ret_z_entry   how violent the return shock must be to count as a cascade
  min_conf      how much agreement the engine wants across its confirmations

Every combination runs with realistic maker fills and full fees, so the numbers
are directly comparable to scripts/maker_fill_realism.py. A separate strategy
instance is needed per ``ret_z_entry`` because a suppressed signal also skips
its cooldown, which changes what fires later — post-filtering one permissive run
would quietly mis-state the alternatives.

The report leads with the concentration columns (share of profit from the best
day, result excluding the best five) because a higher mean bought by even more
tail dependence is not an improvement.

Usage:
    python -m scripts.sweep_entry_threshold [recordings_dir] [--from YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict
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

_BPS = Decimal("10000")

RET_Z_ENTRIES = ["3.0", "4.0", "5.0"]     # 3.0 is live
MIN_CONFIDENCES = ["0", "0.50"]           # 0 = no confidence gate (live)


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
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument(
        "--ret-z",
        default=",".join(RET_Z_ENTRIES),
        help="comma-separated ret_z_entry values (one strategy instance each)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="drop a recording day (repeatable). The point is to check whether a "
             "threshold's advantage survives without the sessions that produced "
             "most of the profit — an edge that only exists on cascade days is a "
             "statement about the regime, not about the threshold.",
    )
    ap.add_argument(
        "--conf",
        default=",".join(MIN_CONFIDENCES),
        help="comma-separated confidence floors. Sweep several to check the "
             "relationship is monotonic rather than a lucky cut point.",
    )
    args = ap.parse_args()
    ret_z_entries = [s.strip() for s in args.ret_z.split(",") if s.strip()]
    min_confidences = [s.strip() for s in args.conf.split(",") if s.strip()]

    settings = HyperliquidSettings.from_env()
    base = DvslaParams.from_settings(settings)

    def cfg() -> SimConfig:
        return SimConfig(
            take_profit_pct=Decimal("0"),
            stop_loss_pct=settings.atr_stop_min_pct,
            time_stop_seconds=Decimal(str(settings.max_hold_seconds)),
            trailing_callback_pct=settings.trailing_callback_pct,
            break_even_trigger_pct=Decimal("0"),
            maker_entry_enabled=True,  # realistic post-only fills throughout
            maker_fee_bps=settings.maker_fee_bps,
            taker_fee_bps=settings.taker_fee_bps,
            maker_fill_timeout_seconds=Decimal(str(settings.max_hold_seconds)),
        )

    # One strategy per ret_z_entry; each feeds every confidence variant.
    strategies = []
    for rz in ret_z_entries:
        params = base.model_copy(update={"ret_z_entry": Decimal(rz)})
        sims = [(Decimal(c), BacktestSimulator(None, cfg())) for c in min_confidences]
        strategies.append((rz, DvslaStrategy(params), sims))

    files = recording_files(args.directory, since=args.since, exclude=args.exclude)
    if not files:
        print("No recordings matched.")
        return 1

    print(f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})"
          + (f"  excluded: {','.join(sorted(args.exclude))}" if args.exclude else ""))
    print(f"Grid: ret_z {','.join(ret_z_entries)} x conf {','.join(min_confidences)}, realistic maker fills")
    print(f"Exit: time_stop={settings.max_hold_seconds}s trail=0 stop="
          f"{float(settings.atr_stop_min_pct) * 100:.2f}% fees="
          f"{float(settings.maker_fee_bps)}/{float(settings.taker_fee_bps)}bps\n")

    async def run() -> None:
        n = 0
        t0 = time.time()
        for path in files:
            for event in replay_file(path):
                n += 1
                price, ts = _price_and_ts(event)
                coin = event.coin.strip().upper() if event.coin else None
                for _, strategy, sims in strategies:
                    if price is not None and coin is not None:
                        for _, sim in sims:
                            sim._marks[coin] = price
                            sim._check_exits(event.coin, price, ts)
                            sim._check_pending_fills(event.coin, price, ts)
                    signal = await strategy.on_market_event(event)
                    if signal is not None:
                        for min_conf, sim in sims:
                            if signal.confidence >= min_conf:
                                sim._open_from_signal(signal)
                if n % 20_000_000 == 0:
                    print(f"  ...{n:,} events ({n / (time.time() - t0):,.0f}/s)")

    asyncio.run(run())
    for _, _, sims in strategies:
        for _, sim in sims:
            sim._close_remaining()

    print(f"{'ret_z':>6} {'conf':>5} {'n':>6} {'fill%':>6} {'win%':>5} {'net$':>9} "
          f"{'bps':>7} {'t':>7} {'best day%':>10} {'ex-top5':>8}")
    print("-" * 80)
    for rz, _, sims in strategies:
        for min_conf, sim in sims:
            r = sim._result
            trades = [t for t in r.trades if t.notional]
            if len(trades) < 2:
                print(f"{rz:>6} {float(min_conf):5.2f} {len(trades):6d}   (too few)")
                continue
            bps = [float(t.net_pnl / t.notional * _BPS) for t in trades]
            nets = [float(t.net_pnl) for t in trades]
            fill = (r.maker_orders_filled / r.maker_orders_placed * 100) if r.maker_orders_placed else 0.0

            byday: dict[object, list[float]] = defaultdict(list)
            for t in trades:
                byday[t.exit_ts.date()].append(float(t.net_pnl / t.notional * _BPS))
            contrib = sorted(
                ((sum(v), d) for d, v in byday.items()), reverse=True
            )
            total = sum(bps)
            best_share = (contrib[0][0] / total * 100) if total else float("nan")
            top5 = {d for _, d in contrib[:5]}
            rest = [b for t, b in zip(trades, bps) if t.exit_ts.date() not in top5]
            ex5 = statistics.mean(rest) if rest else float("nan")

            print(f"{rz:>6} {float(min_conf):5.2f} {len(trades):6d} {fill:6.1f} "
                  f"{sum(1 for x in nets if x > 0) / len(nets) * 100:5.0f} {sum(nets):9.2f} "
                  f"{statistics.mean(bps):7.2f} {_tstat(bps):+7.2f} {best_share:9.0f}% {ex5:+8.2f}")
    print("\nbest day% = share of total bps from the single best session")
    print("ex-top5   = mean bps excluding the five best sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
