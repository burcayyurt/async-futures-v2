"""One-off calibration analysis over recorded events.

Streams the JSONL recording and reports, per symbol:
  * trade throughput (total traded size, trades, span) -> volume-bar threshold
    that yields ~target bars/min
  * open-interest dynamics (update count, distinct values, |OI change| stats)
    to expose the z-score blow-up driver

Usage: python scripts/analyze_recordings.py [path] [target_bars_per_min]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/recordings/events-2026-06-04.jsonl"
    target_bpm = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    trade_sz: dict[str, float] = defaultdict(float)
    trade_n: dict[str, int] = defaultdict(int)
    first_ts: dict[str, datetime] = {}
    last_ts: dict[str, datetime] = {}

    oi_updates: dict[str, int] = defaultdict(int)
    oi_prev: dict[str, float] = {}
    oi_changes: dict[str, int] = defaultdict(int)  # non-zero OI deltas
    oi_zero_deltas: dict[str, int] = defaultdict(int)  # repeated identical OI
    oi_abs_sum: dict[str, float] = defaultdict(float)

    total = 0
    with open(path, "r") as fh:
        for line in fh:
            total += 1
            evt = json.loads(line)
            coin = evt.get("coin")
            if coin is None:
                continue
            kind = evt["kind"]
            ts = _parse_ts(evt["ts"])
            if coin not in first_ts:
                first_ts[coin] = ts
            last_ts[coin] = ts

            if kind == "trade":
                p = evt["payload"]
                trade_sz[coin] += float(p["sz"])
                trade_n[coin] += 1
            elif kind == "asset_ctx":
                p = evt["payload"]
                oi = float(p["open_interest"])
                oi_updates[coin] += 1
                if coin in oi_prev:
                    delta = oi - oi_prev[coin]
                    if delta == 0.0:
                        oi_zero_deltas[coin] += 1
                    else:
                        oi_changes[coin] += 1
                        oi_abs_sum[coin] += abs(delta)
                oi_prev[coin] = oi

    print(f"Total events: {total:,}\n")
    print(f"{'coin':5} {'trades':>9} {'span_min':>9} {'sz/min':>14} "
          f"{'thr@%.1fbpm':>12}".replace('%.1f', f'{target_bpm:.1f}'))
    print("-" * 60)
    recommended: dict[str, float] = {}
    for coin in sorted(trade_sz, key=lambda c: -trade_sz[c]):
        span = (last_ts[coin] - first_ts[coin]).total_seconds() / 60.0
        if span <= 0:
            continue
        sz_per_min = trade_sz[coin] / span
        thr = sz_per_min / target_bpm  # size per bar for target bars/min
        recommended[coin] = thr
        print(f"{coin:5} {trade_n[coin]:>9,} {span:>9.1f} {sz_per_min:>14,.1f} {thr:>12,.0f}")

    print("\nOI dynamics (driver of z-score blow-up):")
    print(f"{'coin':5} {'updates':>9} {'nonzeroΔ':>9} {'zeroΔ':>9} {'%zero':>7} {'avg|Δ|':>14}")
    print("-" * 60)
    for coin in sorted(oi_updates, key=lambda c: -oi_updates[c]):
        upd = oi_updates[coin]
        nz = oi_changes[coin]
        z = oi_zero_deltas[coin]
        denom = nz + z
        pct_zero = (z / denom * 100.0) if denom else 0.0
        avg_abs = (oi_abs_sum[coin] / nz) if nz else 0.0
        print(f"{coin:5} {upd:>9,} {nz:>9,} {z:>9,} {pct_zero:>6.1f}% {avg_abs:>14,.1f}")

    print("\nSuggested symbol_thresholds (Decimal-ready):")
    rounded = {c: _nice(v) for c, v in recommended.items()}
    print(rounded)
    return 0


def _nice(v: float) -> float:
    """Round to 2 significant figures for readable thresholds."""
    if v <= 0:
        return v
    import math
    digits = math.floor(math.log10(v))
    factor = 10 ** (digits - 1)
    return round(v / factor) * factor


if __name__ == "__main__":
    raise SystemExit(main())
