"""Replay recorded events through the calibrated DVSLA engine and report signals.

Quick end-to-end validation that the PR7 calibration (per-symbol volume-bar
thresholds + robust OI z-score) produces signals on real data.

Usage: python scripts/replay_dvsla_signals.py [recordings_dir]
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

from backtest.replay import EventReplayer
from src.core.config import HyperliquidSettings
from src.strategy.dvsla import DvslaParams, DvslaStrategy


async def _run(directory: str):
    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    signals: list = []
    strat = DvslaStrategy(params, on_signal=signals.append)
    n = 0
    for event in EventReplayer.from_directory(directory):
        n += 1
        await strat.on_market_event(event)
    return n, signals


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/recordings"
    n, signals = asyncio.run(_run(directory))
    by_symbol = Counter(s.symbol for s in signals)
    by_side = Counter(s.side.value for s in signals)
    print(f"Events replayed: {n:,}")
    print(f"Signals: {len(signals)}")
    print(f"By symbol: {dict(by_symbol)}")
    print(f"By side: {dict(by_side)}")
    if signals:
        print("\nFirst 15 signals:")
        for s in signals[:15]:
            print(
                f"  {s.timestamp:%H:%M:%S} {s.symbol:5} {s.side.value:5} "
                f"px={s.entry_mark_price} conf={s.confidence} | {s.reason}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
