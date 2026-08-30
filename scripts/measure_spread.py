"""Estimate the bid-ask spread, to find out what a resting entry can be worth.

`MAKER_ENTRY_OFFSET_BPS=2` was deployed on the strength of a backtest and
produced ten orders in a day, all ten rejected: the arrival mark was
bit-identical to the signal mark in eight of them, so a limit two basis points
past it was marketable by construction and could never rest.

That leaves one fact missing. A post-only buy rests only while it stays below
the ask, and the signal mark is near the middle of the book, so the offset a
quote can carry is bounded by the **half-spread**. Nobody here has measured it.
The backtest compared the limit against the next trade print instead, which can
have happened at the bid, so it under-reported crossing — quite possibly the
whole disagreement between it and production.

The recordings have no order book, but trade prints carry the aggressor side:
a ``B`` print lifted the ask, an ``A`` print hit the bid. Two prints of opposite
side, close enough in time that the market cannot have moved much, therefore
bracket the spread. Pairs further apart than ``--max-gap`` are dropped, and
pairs where the "ask" came in below the "bid" are counted separately rather than
silently discarded: a high share of those means the estimate is being swamped by
drift and should not be trusted.

This is a floor on the true spread, not the spread itself — the touch may be
wider than the last print on each side. It is enough to answer whether a 2 bps
offset was ever restable.

Usage:
    python -m scripts.measure_spread [dir] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                                     [--max-gap 1.0]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from backtest.recording_paths import recording_files
from backtest.replay import replay_file
from src.exchange.hyperliquid_ws import EventKind, TradePayload

_BPS = 10_000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument("--to", dest="until", default=None)
    ap.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="seconds allowed between the two prints of a pair. Wider windows "
             "measure drift as much as spread.",
    )
    args = ap.parse_args()

    files = recording_files(args.directory, since=args.since, until=args.until)
    if not files:
        print("No recordings matched.")
        return 1
    print(f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})  "
          f"max gap {args.max_gap}s\n")

    # coin -> last print price and timestamp, per aggressor side.
    last: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    spreads: dict[str, list[float]] = defaultdict(list)
    crossed: dict[str, int] = defaultdict(int)

    for path in files:
        for event in replay_file(path):
            if event.kind != EventKind.TRADE or event.coin is None:
                continue
            payload = event.payload
            if not isinstance(payload, TradePayload) or payload.side not in ("A", "B"):
                continue
            coin = event.coin.strip().upper()
            px = float(payload.px)
            ts = event.ts.timestamp()
            book = last[coin]
            book[payload.side] = (px, ts)

            other = "A" if payload.side == "B" else "B"
            if other not in book:
                continue
            other_px, other_ts = book[other]
            if abs(ts - other_ts) > args.max_gap:
                continue
            ask = px if payload.side == "B" else other_px
            bid = other_px if payload.side == "B" else px
            mid = (ask + bid) / 2
            if mid <= 0:
                continue
            if ask <= bid:
                # The market moved between the two prints; this pair says
                # nothing about the spread.
                crossed[coin] += 1
                continue
            spreads[coin].append((ask - bid) / mid * _BPS)

    print(f"{'coin':>6} {'pairs':>7} {'drift%':>7} {'p25':>7} {'medyan':>8} {'p75':>7} "
          f"{'yari-spread':>12}")
    print("-" * 62)
    all_medians = []
    for coin in sorted(spreads, key=lambda c: -len(spreads[c])):
        xs = sorted(spreads[coin])
        if len(xs) < 50:
            continue
        drift = crossed[coin] / (crossed[coin] + len(xs)) * 100
        med = statistics.median(xs)
        all_medians.append(med)
        q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
        print(f"{coin:>6} {len(xs):7d} {drift:6.0f}% {q(0.25):7.2f} {med:8.2f} "
              f"{q(0.75):7.2f} {med / 2:11.2f}")

    if all_medians:
        overall = statistics.median(all_medians)
        print(f"\nSembollerin medyan spread'inin medyani: {overall:.2f} bps")
        print(f"Dinlenebilir azami ofset (yari-spread): ~{overall / 2:.2f} bps")
        print("Denenen ofset 2.00 bps -> "
              + ("bu esigin ALTINDA, dinlenebilirdi"
                 if 2.0 < overall / 2
                 else "bu esigin USTUNDE: emir marketable olur, reddedilir"))
    print("\nBu bir ALT SINIR: gercek touch, her iki taraftaki son print'ten")
    print("daha genis olabilir. Drift% yuksekse tahmin guvenilmez.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
