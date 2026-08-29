"""Exit-agnostic edge test for DVSLA signals via forward-return / MFE-MAE.

A take-profit/stop sweep can only tune an exit; it cannot tell you whether the
*entry* has directional edge. This does: it replays the recording once, runs the
strategy, and for every signal tracks the realised price path in the **faded
direction** for a set of horizons. If the average favourable move barely exceeds
the average adverse move (and round-trip costs), there is no edge to harvest and
no exit can rescue it.

Sign convention: returns are expressed in the *signal's* direction (LONG => +up,
SHORT => +down), in percent of entry price. Positive = the fade worked.

Usage:
    python scripts/analyze_signal_edge.py [recordings_dir] [--days N]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime
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

HORIZONS = (30, 60, 180, 300, 600)
MAX_H = max(HORIZONS)


@dataclass(slots=True)
class _Active:
    side: SignalSide
    entry_px: Decimal
    entry_ts: datetime
    mfe: float = 0.0  # best favourable move (%), >= 0
    mae: float = 0.0  # worst adverse move (%), <= 0
    horizon_ret: dict[int, float] = field(default_factory=dict)


def _price_and_ts(event: MarketEvent):
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _signed_ret(side: SignalSide, entry: Decimal, px: Decimal) -> float:
    r = float((px - entry) / entry) * 100.0
    return r if side == SignalSide.LONG else -r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    strategy = DvslaStrategy(DvslaParams.from_settings(settings))

    files = recording_files(args.directory)
    if args.days:
        files = files[-args.days:]
    print(f"Files: {[f.name for f in files]}")

    active: dict[str, list[_Active]] = {}
    done: list[_Active] = []

    async def run() -> int:
        n = 0
        for path in files:
            for event in replay_file(path):
                n += 1
                price, ts = _price_and_ts(event)
                if price is not None and event.coin is not None and price > 0:
                    key = event.coin.strip().upper()
                    lst = active.get(key)
                    if lst:
                        keep = []
                        for a in lst:
                            elapsed = (ts - a.entry_ts).total_seconds()
                            r = _signed_ret(a.side, a.entry_px, price)
                            a.mfe = max(a.mfe, r)
                            a.mae = min(a.mae, r)
                            for h in HORIZONS:
                                if h not in a.horizon_ret and elapsed >= h:
                                    a.horizon_ret[h] = r
                            if elapsed >= MAX_H:
                                done.append(a)
                            else:
                                keep.append(a)
                        active[key] = keep
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    key = signal.symbol.strip().upper()
                    active.setdefault(key, []).append(
                        _Active(
                            side=signal.side,
                            entry_px=signal.entry_mark_price,
                            entry_ts=signal.timestamp,
                        )
                    )
        return n

    n = asyncio.run(run())
    for lst in active.values():
        done.extend(lst)
    print(f"Replayed {n:,} events | signals tracked: {len(done)}\n")

    if not done:
        print("No signals.")
        return 0

    def stats(vals: list[float]) -> str:
        if not vals:
            return "n=0"
        vals_sorted = sorted(vals)
        med = statistics.median(vals_sorted)
        mean = statistics.fmean(vals_sorted)
        return f"mean={mean:+.3f}% med={med:+.3f}%"

    mfe = [a.mfe for a in done]
    mae = [a.mae for a in done]
    print(f"MFE (best favourable, within {MAX_H}s): {stats(mfe)}")
    print(f"MAE (worst adverse,    within {MAX_H}s): {stats(mae)}")
    print(f"MFE/|MAE| ratio (mean): {statistics.fmean(mfe) / max(abs(statistics.fmean(mae)), 1e-9):.2f}\n")

    print(f"{'horizon':>8} {'n':>4} {'mean ret%':>10} {'med ret%':>9} {'win%':>6}")
    print("-" * 42)
    for h in HORIZONS:
        rets = [a.horizon_ret[h] for a in done if h in a.horizon_ret]
        if not rets:
            print(f"{h:8d} {0:4d}")
            continue
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        print(f"{h:8d} {len(rets):4d} {statistics.fmean(rets):+10.4f} "
              f"{statistics.median(rets):+9.4f} {win:6.0f}")

    # Round-trip cost reference: maker in + taker out + slippage ~ 8 bps = 0.08%.
    print("\nRound-trip cost reference: ~0.08% price (8 bps). "
          "Mean favourable move must clear this to have edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
