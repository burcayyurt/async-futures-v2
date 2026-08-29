"""Measure whether a DVSLA signal predicts anything at all.

Exit tuning cannot rescue an entry with no edge, so this strips the exit layer
away entirely: for every signal, record the raw forward return at fixed horizons
and compare it against a baseline of random entries drawn from the same price
series. If the signal has predictive power its forward returns beat the baseline
by more than sampling noise; if they do not, no stop/trail/target arrangement
will produce a profitable system.

Returns are direction-adjusted (a short's forward return is negated) and are
reported before fees, so the fee hurdle (~0.060% round trip) is the bar a mean
must clear to be worth trading at all.

Usage:
    python -m scripts.signal_forward_returns [recordings_dir] [--days N]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
from collections import defaultdict, deque
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

HORIZONS = [30, 60, 300, 900]  # seconds; override with --horizons
ROUND_TRIP_FEE_PCT = 0.060  # maker in + taker out, for reference
BASELINE_RATE = 0.0002  # fraction of price ticks sampled as random entries
CONFIDENCE_HORIZON = 60  # where the signal's edge peaks
CONFIDENCE_EDGES = (0.35, 0.45, 0.55)


def _confidence_bucket(conf: float) -> str:
    if conf < CONFIDENCE_EDGES[0]:
        return f"<{CONFIDENCE_EDGES[0]:.2f}"
    for lo, hi in zip(CONFIDENCE_EDGES, CONFIDENCE_EDGES[1:]):
        if conf < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f">={CONFIDENCE_EDGES[-1]:.2f}"


class _Pending:
    __slots__ = ("coin", "entry_px", "ts", "is_long", "remaining", "confidence")

    def __init__(
        self, coin: str, entry_px: float, ts: datetime, is_long: bool, confidence: float = 0.0
    ) -> None:
        self.coin = coin
        self.entry_px = entry_px
        self.ts = ts
        self.is_long = is_long
        self.remaining = list(HORIZONS)
        self.confidence = confidence


def _price_and_ts(event: MarketEvent) -> tuple[float | None, datetime | None]:
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return float(event.payload.px), event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return float(event.payload.mark_px), event.ts
    return None, None


def _summarize(label: str, samples: dict[int, list[float]]) -> dict[int, tuple]:
    out = {}
    for h in HORIZONS:
        vals = samples.get(h, [])
        if len(vals) < 2:
            out[h] = (len(vals), float("nan"), float("nan"), float("nan"))
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        t = mean / (sd / len(vals) ** 0.5) if sd > 0 else float("nan")
        out[h] = (len(vals), mean, sd, t)
    return out


def main() -> int:
    global HORIZONS
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="drop a recording day. Every other study here has had to ask "
             "whether a result survives without the sessions that produced most "
             "of it; a forward-return mean is no different.",
    )
    ap.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="seconds to measure the forward return over. Widening these asks "
             "whether the signal says anything on a timescale where the ~6 bps "
             "round-trip fee is a small part of the move rather than most of it.",
    )
    args = ap.parse_args()
    HORIZONS = sorted(int(h) for h in args.horizons.split(",") if h.strip())

    rng = random.Random(args.seed)
    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    strategy = DvslaStrategy(params)

    signal_samples: dict[int, list[float]] = defaultdict(list)
    baseline_samples: dict[int, list[float]] = defaultdict(list)
    # Forward return at the peak horizon, split by the signal's own confidence.
    # If confidence carries information, filtering on it is the only lever that
    # raises edge *per trade* — and the fee hurdle is charged per trade.
    by_confidence: dict[str, list[float]] = defaultdict(list)
    # (confidence, return) pairs so a cutoff can be swept afterwards. Buckets
    # alone cannot tell a real threshold from one the analyst happened to pick.
    conf_pairs: list[tuple[float, float]] = []
    pending_signal: deque[_Pending] = deque()
    pending_baseline: deque[_Pending] = deque()
    signal_count = 0

    def _resolve(queue: deque[_Pending], bucket: dict[int, list[float]],
                 coin: str, price: float, ts: datetime) -> None:
        # Queue is time-ordered; walk from the front while entries have matured.
        keep: deque[_Pending] = deque()
        while queue:
            p = queue.popleft()
            if p.coin != coin:
                keep.append(p)
                continue
            matured = [h for h in p.remaining if ts >= p.ts + timedelta(seconds=h)]
            for h in matured:
                ret = (price - p.entry_px) / p.entry_px * 100.0
                directional = ret if p.is_long else -ret
                bucket[h].append(directional)
                if h == CONFIDENCE_HORIZON and p.confidence:
                    by_confidence[_confidence_bucket(p.confidence)].append(directional)
                    conf_pairs.append((p.confidence, directional))
                p.remaining.remove(h)
            if p.remaining:
                keep.append(p)
        queue.extend(keep)

    files = recording_files(args.directory, exclude=args.exclude)
    if args.days:
        files = files[-args.days:]
    print(f"Files: {len(files)}   horizons: {HORIZONS}s   invert={params.invert}"
          + (f"   excluded: {','.join(sorted(args.exclude))}" if args.exclude else ""))

    async def run() -> None:
        nonlocal signal_count
        for path in files:
            for event in replay_file(path):
                price, ts = _price_and_ts(event)
                if price is not None and ts is not None and event.coin is not None:
                    coin = event.coin.strip().upper()
                    _resolve(pending_signal, signal_samples, coin, price, ts)
                    _resolve(pending_baseline, baseline_samples, coin, price, ts)
                    if rng.random() < BASELINE_RATE:
                        pending_baseline.append(
                            _Pending(coin, price, ts, rng.random() < 0.5)
                        )
                signal = await strategy.on_market_event(event)
                if signal is not None and ts is not None:
                    signal_count += 1
                    pending_signal.append(
                        _Pending(
                            signal.symbol.strip().upper(),
                            float(signal.entry_mark_price),
                            ts,
                            signal.side == SignalSide.LONG,
                            float(signal.confidence),
                        )
                    )

    asyncio.run(run())

    sig = _summarize("signal", signal_samples)
    base = _summarize("baseline", baseline_samples)

    print(f"\nSignals fired: {signal_count}")
    print(f"Fee hurdle to beat: {ROUND_TRIP_FEE_PCT:.3f}% round trip\n")
    print(f"{'horizon':>8} {'n':>6} {'mean%':>9} {'t':>7} | {'base n':>7} {'base mean%':>11} "
          f"| {'edge vs base':>13}")
    print("-" * 78)
    for h in HORIZONS:
        n, mean, sd, t = sig[h]
        bn, bmean, bsd, bt = base[h]
        edge = mean - bmean if mean == mean and bmean == bmean else float("nan")
        print(f"{h:7d}s {n:6d} {mean:+9.4f} {t:+7.2f} | {bn:7d} {bmean:+11.4f} "
              f"| {edge:+13.4f}")

    print(f"\n--- forward return at {CONFIDENCE_HORIZON}s, split by signal confidence ---")
    print(f"{'confidence':>12} {'n':>6} {'mean%':>9} {'t':>7}  {'vs fee hurdle':>14}")
    print("-" * 56)
    order = [f"<{CONFIDENCE_EDGES[0]:.2f}"]
    order += [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(CONFIDENCE_EDGES, CONFIDENCE_EDGES[1:])]
    order += [f">={CONFIDENCE_EDGES[-1]:.2f}"]
    for label in order:
        vals = by_confidence.get(label, [])
        if len(vals) < 2:
            print(f"{label:>12} {len(vals):6d}      (too few)")
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        t = mean / (sd / len(vals) ** 0.5) if sd > 0 else float("nan")
        margin = mean - ROUND_TRIP_FEE_PCT
        print(f"{label:>12} {len(vals):6d} {mean:+9.4f} {t:+7.2f}  {margin:+14.4f}")

    # A genuine threshold shows as a step; an artefact of the chosen bucket edges
    # shows as a smooth slope with no particular cutoff standing out.
    print(f"\n--- cumulative: keep only signals with confidence >= X ({CONFIDENCE_HORIZON}s) ---")
    print(f"{'cutoff':>7} {'kept':>6} {'% kept':>7} {'mean%':>9} {'t':>7} {'net of fee':>11}")
    print("-" * 54)
    total = len(conf_pairs)
    for cutoff in (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        kept = [r for c, r in conf_pairs if c >= cutoff]
        if len(kept) < 2:
            print(f"{cutoff:7.2f} {len(kept):6d}      (too few)")
            continue
        mean = statistics.mean(kept)
        sd = statistics.stdev(kept)
        t = mean / (sd / len(kept) ** 0.5) if sd > 0 else float("nan")
        share = 100.0 * len(kept) / total if total else 0.0
        print(
            f"{cutoff:7.2f} {len(kept):6d} {share:6.1f}% {mean:+9.4f} {t:+7.2f} "
            f"{mean - ROUND_TRIP_FEE_PCT:+11.4f}"
        )

    print("\nRead: a signal only pays if its mean forward return clears the fee")
    print("hurdle AND beats the random baseline by more than noise (|t| >= 2).")
    print("If the confidence buckets separate, filtering raises edge per trade —")
    print("the only lever that helps, since the fee is charged per trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
