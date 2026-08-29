"""Price the round trip: how far does the market move while an order is in flight?

The bot runs from Istanbul and the Hyperliquid origin answers in ~258 ms, so a
post-only entry is placed against a mark the market has already left behind. The
question that decides whether closer hosting is worth paying for is not the ping
figure — it is how many basis points the price travels during that window, in the
direction the signal just called.

For a momentum entry the drift is pure cost either way. If price keeps running,
the resting buy sits below the market and never fills, and the trades that do
fill are the ones price came back to — the adverse-selection channel. If price
snaps back, the order fills at a price that is already stale. Both are measured
here as signed drift in the signal's own direction, so positive means the market
left without us.

The recordings carry receive timestamps from this machine, so the measured drift
is what a bot hosted *here* suffers. Latency to the exchange cannot be removed
from the recording, which means this is a lower bound: the true cost also
includes the delay before the signal-forming tick reached us in the first place.

Usage:
    python -m scripts.latency_cost [recordings_dir] [--from YYYY-MM-DD]
                                   [--horizons 130,260,500,1000]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal

from backtest.replay import recording_files, replay_file
from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.dvsla import DvslaParams, DvslaStrategy
from src.strategy.signals import SignalSide

_BPS = Decimal("10000")


def _price_and_ts(event: MarketEvent) -> tuple[Decimal | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


class _Pending:
    """One signal waiting for the marks that arrive during its flight window."""

    __slots__ = ("coin", "sign", "px0", "ts0", "deadline", "seen")

    def __init__(self, coin: str, sign: int, px0: Decimal, ts0: datetime, horizons: list[float]):
        self.coin = coin
        self.sign = sign
        self.px0 = px0
        self.ts0 = ts0
        self.deadline = ts0 + timedelta(milliseconds=max(horizons))
        # Last mark seen at or before each horizon — the price an order placed
        # that late would have been quoted against.
        self.seen: dict[float, Decimal] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument(
        "--horizons",
        default="15,30,60,130,260,500",
        help="milliseconds of in-flight delay to price (comma separated). 260 is "
             "the measured round trip from Istanbul, ~15 what a Tokyo host would "
             "see. Sweeping the short end answers the question the ping figure "
             "cannot: whether the move is spread across the window (closer "
             "hosting recovers most of it) or already finished in the first "
             "instants (closer hosting recovers nothing).",
    )
    args = ap.parse_args()
    horizons = sorted(float(h) for h in args.horizons.split(",") if h.strip())

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))
    floor = settings.dvsla_min_confidence

    files = recording_files(args.directory, since=args.since)
    if not files:
        print("No recordings matched.")
        return 1

    print(f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})")
    print(f"Confidence floor {float(floor):.2f} (live), horizons {horizons} ms\n")

    rows: list[dict[float, float]] = []
    pending: list[_Pending] = []

    async def run() -> None:
        nonlocal pending
        for path in files:
            for event in replay_file(path):
                price, ts = _price_and_ts(event)
                coin = event.coin.strip().upper() if event.coin else None

                if price is not None and ts is not None and coin is not None:
                    still: list[_Pending] = []
                    for p in pending:
                        if p.coin == coin:
                            age_ms = (ts - p.ts0).total_seconds() * 1000
                            for h in horizons:
                                if age_ms <= h:
                                    p.seen[h] = price
                        if ts <= p.deadline:
                            still.append(p)
                        elif p.px0:
                            rows.append(
                                {
                                    h: float((px - p.px0) / p.px0 * _BPS) * p.sign
                                    for h, px in p.seen.items()
                                }
                            )
                    pending = still

                signal = await strategy.on_market_event(event)
                if signal is not None and signal.confidence >= floor:
                    sign = 1 if signal.side == SignalSide.LONG else -1
                    pending.append(
                        _Pending(
                            signal.symbol.strip().upper(),
                            sign,
                            signal.entry_mark_price,
                            signal.timestamp,
                            horizons,
                        )
                    )

    asyncio.run(run())

    def table(sample, title):
        print()
        print(f"{title}  (n={len(sample)})")
        print(f"{'gecikme':>9} {'n':>6} {'ort bps':>9} {'medyan':>8} {'>0 %':>6} "
              f"{'p75':>7} {'p90':>7} {'sd':>7} {'t':>7}")
        print('-' * 72)
        for h in horizons:
            xs = [r[h] for r in sample if h in r]
            if len(xs) < 2:
                print(f"{h:8.0f}ms {len(xs):6d}   (too few)")
                continue
            xs_sorted = sorted(xs)
            q = lambda f: xs_sorted[min(len(xs_sorted) - 1, int(f * len(xs_sorted)))]
            sd = statistics.stdev(xs)
            t = statistics.mean(xs) / (sd / len(xs) ** 0.5) if sd > 0 else float('nan')
            print(f"{h:8.0f}ms {len(xs):6d} {statistics.mean(xs):+9.2f} "
                  f"{statistics.median(xs):+8.2f} "
                  f"{sum(1 for x in xs if x > 0) / len(xs) * 100:5.0f}% "
                  f"{q(0.75):+7.2f} {q(0.90):+7.2f} {sd:7.1f} {t:+7.2f}")

    # Pooling per horizon compares different signals: a 15 ms column exists only
    # for signals whose next tick arrived that fast, which are the busiest — and
    # busiest means largest move. The common subset holds the signal set fixed,
    # so the difference between columns is latency and nothing else.
    table(rows, 'TUM SINYALLER (ufuklar farkli alt kumeler - kiyaslanamaz)')
    common = [r for r in rows if all(h in r for h in horizons)]
    table(common, 'ORTAK ALT KUME (her ufukta ayni sinyaller - kiyaslanabilir)')

    print("\nPozitif = sinyalin yonunde kacan fiyat: emrimiz o kadar geride kaliyor.")
    print("Bu makinenin kayitlarindan olculdu, yani zaten gecikmis bir goruntu;")
    print("gercek maliyet bundan buyuk olabilir, kucuk olamaz.")
    print("Karari ORTAK ALT KUME tablosuna gore ver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
