"""Per-symbol economics: does each symbol's signal edge cover its own costs?

The portfolio-level answer has been measured repeatedly and is ~0. That average
hides two things this script separates, because both are symbol-specific.

**Cost.** A post-only entry rests at the mark and pays no spread, but the exit
is a taker order and crosses to the far touch. So the round trip costs
``maker_fee + taker_fee + half_spread`` -- and the half-spread ranges from
fractions of a basis point on BTC to several bps on the thin alts. A signal
worth +5 bps is a business on one symbol and a donation on another.

**Bar scale.** ``dvsla_symbol_thresholds`` is denominated in coin units, so the
notional a bar represents varies by orders of magnitude across the watchlist.
Bars that are too small in notional close on near-identical prices, collapse the
rolling variance, and drive ``ret_z`` into the clamp ceiling -- where the
artefact gate discards them. This reports the notional each symbol's bar
actually carries and how often that symbol trips the gate, so the thresholds can
be re-scaled against something measured instead of guessed.

The edge here is measured at the **signal** level: entry at the signal mark with
certainty, exit on the clock. It is an upper bound. Realised trading is lower --
unfilled post-only orders, adverse selection on the ones that do fill, and the
five-slot capacity limit all subtract from it (see sweep_time_stop.py). A symbol
that fails to clear its costs *here* cannot be rescued downstream.

Spread is estimated from opposite-aggressor trade prints, as in
``scripts/measure_spread.py``: a ``B`` print lifted the ask and an ``A`` print
hit the bid, so two prints of opposite side within ``--max-gap`` bracket the
spread. It is a floor on the true spread, not the spread itself.

Usage:
    python -m scripts.symbol_economics [dir] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                                       [--horizon 60] [--max-gap 1.0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from backtest.recording_paths import recording_files
from backtest.replay import replay_file
from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.dvsla import DvslaParams, DvslaStrategy
from src.strategy.signals import SignalSide

_BPS = 10_000.0
_SKIP_RE = re.compile(r"DVSLA skip ([A-Z0-9]+): ret_z=")


@dataclass(slots=True)
class _Pending:
    side: SignalSide
    entry_px: Decimal
    entry_ts: datetime
    conf: float
    day: str
    ret_bps: float | None = None


@dataclass
class _Sym:
    bars: int = 0
    bar_notional: list[float] = field(default_factory=list)
    bar_seconds: list[float] = field(default_factory=list)
    bar_trades: list[int] = field(default_factory=list)
    # A bar whose open and close are the same price could not move the market it
    # is supposed to be measuring. It still enters the rolling variance window
    # as a zero, which is how the variance collapses.
    flat_bars: int = 0
    artefacts: int = 0
    spreads: list[float] = field(default_factory=list)
    drift_pairs: int = 0
    signals: list[_Pending] = field(default_factory=list)


class _SkipCounter(logging.Handler):
    """Count artefact rejections by reading the strategy's own log records.

    Recomputing the gate here would duplicate the decision it is meant to
    observe; counting the log line cannot drift from the code that emits it.
    """

    def __init__(self, syms: dict[str, _Sym]) -> None:
        super().__init__(level=logging.INFO)
        self._syms = syms

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        match = _SKIP_RE.search(message)
        if match:
            self._syms[match.group(1)].artefacts += 1


def _price_and_ts(event: MarketEvent):
    if event.coin is None:
        return None, None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return event.payload.px, event.ts
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return event.payload.mark_px, event.ts
    return None, None


def _signed_bps(side: SignalSide, entry: Decimal, px: Decimal) -> float:
    r = float((px - entry) / entry) * _BPS
    return r if side == SignalSide.LONG else -r


def _tstat(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 2:
        return (statistics.fmean(xs) if xs else 0.0), 0.0, 0.0
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    if sd == 0:
        return mean, 0.0, 0.0
    return mean, sd, mean / (sd / math.sqrt(n))


def _threshold_for(params: DvslaParams, coin: str) -> float:
    getter = getattr(params, "threshold_for", None)
    if callable(getter):
        return float(getter(coin))
    return float(params.symbol_thresholds.get(coin, params.volume_bar_threshold))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--from", dest="since", default=None)
    ap.add_argument("--to", dest="until", default=None)
    ap.add_argument("--horizon", type=int, default=60, help="exit clock, seconds")
    ap.add_argument("--max-gap", type=float, default=1.0, help="spread pair window, s")
    ap.add_argument(
        "--dump",
        default=None,
        metavar="PATH",
        help="write the per-signal rows and per-symbol aggregates to JSON. "
             "Re-slicing a window (dropping tail days, moving the confidence "
             "floor) then costs nothing, instead of replaying the recordings "
             "again -- and the numbers in a report can be traced back to the "
             "rows they came from.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="drop signals opened on this date from the economics table. Bars "
             "and strategy state still run through the day, so continuity of "
             "the rolling windows is preserved.",
    )
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    min_conf = float(settings.dvsla_min_confidence)
    maker_fee = float(settings.maker_fee_bps)
    taker_fee = float(settings.taker_fee_bps)

    files = recording_files(args.directory, since=args.since, until=args.until)
    if not files:
        print("No recordings matched.")
        return 1
    print(f"Files: {len(files)}  ({files[0].stem} -> {files[-1].stem})")
    print(
        f"Horizon {args.horizon}s | conf floor {min_conf:.2f} | "
        f"fees maker {maker_fee} / taker {taker_fee} bps\n"
    )

    syms: dict[str, _Sym] = defaultdict(_Sym)
    handler = _SkipCounter(syms)
    dvsla_log = logging.getLogger("src.strategy.dvsla")
    dvsla_log.addHandler(handler)
    dvsla_log.setLevel(logging.INFO)
    dvsla_log.propagate = False

    strategy = DvslaStrategy(params)

    # Observe every closed bar without touching the decision it feeds.
    original_on_bar_close = strategy._on_bar_close

    def observed_on_bar_close(coin, state, bar, now):
        s = syms[coin.strip().upper()]
        s.bars += 1
        s.bar_notional.append(float(bar.volume * bar.close_px))
        s.bar_seconds.append((bar.close_ts - bar.open_ts).total_seconds())
        s.bar_trades.append(bar.trade_count)
        if bar.open_px == bar.close_px:
            s.flat_bars += 1
        return original_on_bar_close(coin, state, bar, now)

    strategy._on_bar_close = observed_on_bar_close  # type: ignore[method-assign]

    # coin -> aggressor side -> (price, ts), for the spread estimate.
    last_print: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    open_signals: dict[str, list[_Pending]] = defaultdict(list)

    async def run() -> int:
        n = 0
        for path in files:
            for event in replay_file(path):
                n += 1
                coin = event.coin.strip().upper() if event.coin else None

                price, ts = _price_and_ts(event)
                if coin is not None and price is not None and price > 0:
                    waiting = open_signals.get(coin)
                    if waiting:
                        still_open = []
                        for p in waiting:
                            if (ts - p.entry_ts).total_seconds() >= args.horizon:
                                p.ret_bps = _signed_bps(p.side, p.entry_px, price)
                            else:
                                still_open.append(p)
                        open_signals[coin] = still_open

                if (
                    coin is not None
                    and event.kind == EventKind.TRADE
                    and isinstance(event.payload, TradePayload)
                    and event.payload.side in ("A", "B")
                ):
                    payload = event.payload
                    px = float(payload.px)
                    tstamp = event.ts.timestamp()
                    book = last_print[coin]
                    book[payload.side] = (px, tstamp)
                    other = "A" if payload.side == "B" else "B"
                    if other in book:
                        other_px, other_ts = book[other]
                        if abs(tstamp - other_ts) <= args.max_gap:
                            ask = px if payload.side == "B" else other_px
                            bid = other_px if payload.side == "B" else px
                            mid = (ask + bid) / 2
                            if mid > 0:
                                if ask <= bid:
                                    syms[coin].drift_pairs += 1
                                else:
                                    syms[coin].spreads.append(
                                        (ask - bid) / mid * _BPS
                                    )

                signal = await strategy.on_market_event(event)
                if signal is not None:
                    key = signal.symbol.strip().upper()
                    pending = _Pending(
                        side=signal.side,
                        entry_px=signal.entry_mark_price,
                        entry_ts=signal.timestamp,
                        conf=float(signal.confidence),
                        day=signal.timestamp.date().isoformat(),
                    )
                    syms[key].signals.append(pending)
                    open_signals[key].append(pending)
        return n

    n_events = asyncio.run(run())
    dvsla_log.removeHandler(handler)
    print(f"Replayed {n_events:,} events\n")

    if args.dump:
        payload = {
            "window": [files[0].stem, files[-1].stem],
            "horizon": args.horizon,
            "min_conf": min_conf,
            "maker_fee_bps": maker_fee,
            "taker_fee_bps": taker_fee,
            "symbols": {
                coin: {
                    "bars": sym.bars,
                    "artefacts": sym.artefacts,
                    "bar_notional_median": (
                        statistics.median(sym.bar_notional) if sym.bar_notional else None
                    ),
                    "bar_seconds_median": (
                        statistics.median(sym.bar_seconds) if sym.bar_seconds else None
                    ),
                    "bar_trades_median": (
                        statistics.median(sym.bar_trades) if sym.bar_trades else None
                    ),
                    "flat_bars": sym.flat_bars,
                    "threshold": _threshold_for(params, coin),
                    "spread_pairs": len(sym.spreads),
                    "spread_median": (
                        statistics.median(sym.spreads) if sym.spreads else None
                    ),
                    "drift_pairs": sym.drift_pairs,
                    "signals": [
                        {
                            "day": p.day,
                            "conf": round(p.conf, 4),
                            "ret_bps": (
                                None if p.ret_bps is None else round(p.ret_bps, 4)
                            ),
                        }
                        for p in sym.signals
                    ],
                }
                for coin, sym in syms.items()
            },
        }
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"Dump: {args.dump}\n")

    excluded = set(args.exclude)
    if excluded:
        print(f"Excluded days: {', '.join(sorted(excluded))}")

    rows = []
    for coin, s in syms.items():
        traded = [
            p
            for p in s.signals
            if p.conf >= min_conf and p.ret_bps is not None and p.day not in excluded
        ]
        half = statistics.median(s.spreads) / 2 if len(s.spreads) >= 50 else None
        vals: list[float] = [p.ret_bps for p in traded]  # type: ignore[misc]
        mean, sd, t = _tstat(vals)
        med = statistics.median(vals) if vals else 0.0
        cost = maker_fee + taker_fee + half if half is not None else None
        rows.append((coin, s, traded, half, mean, med, t, cost))

    # Rank on the median, not the mean: the mean is what a cascade session can
    # buy on its own, the median is what the symbol does on an ordinary day.
    rows.sort(key=lambda r: (r[7] is None, -(r[5] - r[7]) if r[7] is not None else 0))

    print("=== 1. Sembol basina ekonomi (sinyal seviyesi, conf >= floor) ===\n")
    print(
        f"{'coin':>6} {'n':>5} {'ort':>8} {'medyan':>8} {'t':>6} {'yari-sp':>8} "
        f"{'maliyet':>8} {'net-ort':>8} {'net-med':>8}  {'karar':<14}"
    )
    print("-" * 92)
    for coin, s, traded, half, mean, med, t, cost in rows:
        if cost is None:
            print(
                f"{coin:>6} {len(traded):5d} {mean:+8.2f} {med:+8.2f} {t:+6.2f} "
                f"{'-':>8} {'-':>8} {'-':>8} {'-':>8}  spread yok"
            )
            continue
        net_mean = mean - cost
        net_med = med - cost
        if len(traded) < 20:
            verdict = "n yetersiz"
        elif net_med > 0 and net_mean > 0 and t > 2:
            verdict = "TASIYOR"
        elif net_mean > 0:
            verdict = "kuyruga bagli"
        else:
            verdict = "tasimiyor"
        print(
            f"{coin:>6} {len(traded):5d} {mean:+8.2f} {med:+8.2f} {t:+6.2f} "
            f"{half:8.2f} {cost:8.2f} {net_mean:+8.2f} {net_med:+8.2f}  "
            f"{verdict:<14}"
        )

    print(
        f"\n  edge = sinyal yonunde {args.horizon}s ileri getiri (bps), "
        f"kesin giris varsayar"
    )
    print(
        f"  maliyet = maker {maker_fee} + taker {taker_fee} + yari-spread "
        f"(cikis touch'i gecer)"
    )
    print("  net > 0 sadece GEREK sart: gerceklesen islem, dolum kaybi ve ters")
    print("  secilim yuzunden bunun altinda kalir.")
    print("  net-med <= 0 < net-ort ise kar birkac kuyruk gunune bagli demektir:")
    print("  tipik sinyal para kaybediyor, ortalamayi birkac seans tasiyor.")

    print("\n\n=== 1b. Gun bazinda (tum semboller, conf >= floor) ===\n")
    per_day: dict[str, list[float]] = defaultdict(list)
    for s in syms.values():
        for p in s.signals:
            if p.conf >= min_conf and p.ret_bps is not None:
                per_day[p.day].append(p.ret_bps)
    day_means = [(d, statistics.fmean(v), len(v)) for d, v in sorted(per_day.items())]
    if day_means:
        grand = statistics.fmean([m for _, m, _ in day_means])
        typical = statistics.median([abs(m) for _, m, _ in day_means])
        print("  En buyuk sapmali 10 gun:")
        for day, m, k in sorted(day_means, key=lambda kv: -abs(kv[1]))[:10]:
            flag = "  <-- kuyruk gunu" if abs(m) > 4 * typical else ""
            print(f"  {day}  n={k:4d}  ort={m:+9.2f} bps{flag}")
        print(f"\n  Gunluk ortalamalarin ortalamasi: {grand:+.2f} bps "
              f"({len(day_means)} gun)")
        print(f"  Gunluk |ortalama| medyani:       {typical:.2f} bps")
        print("  Kuyruk gunlerini --exclude ile dusurup tabloyu tekrar okuyun.")

    print("\n\n=== 2. Bar olcegi ve artefact orani ===\n")
    print(
        f"{'coin':>6} {'esik':>9} {'bar':>9} {'medyan $':>10} {'medyan sn':>10} "
        f"{'islem':>6} {'duz%':>6} {'artefact':>9} {'oran':>7}"
    )
    print("-" * 82)
    scale = []
    for coin, s in sorted(syms.items(), key=lambda kv: -kv[1].bars):
        if s.bars == 0:
            continue
        notional = statistics.median(s.bar_notional)
        secs = statistics.median(s.bar_seconds)
        scale.append((coin, notional))
        print(
            f"{coin:>6} {_threshold_for(params, coin):9.0f} {s.bars:9,d} "
            f"{notional:10,.0f} {secs:10.2f} "
            f"{statistics.median(s.bar_trades):6.0f} "
            f"{s.flat_bars / s.bars * 100:5.1f}% {s.artefacts:9,d} "
            f"{s.artefacts / s.bars * 100:6.2f}%"
        )

    if len(scale) >= 2:
        vals = [v for _, v in scale]
        lo = min(scale, key=lambda kv: kv[1])
        hi = max(scale, key=lambda kv: kv[1])
        print(f"\n  Bar notional medyani: {statistics.median(vals):,.0f} $")
        print(
            f"  En kucuk {lo[0]} {lo[1]:,.0f} $  <->  en buyuk {hi[0]} "
            f"{hi[1]:,.0f} $  = {hi[1] / max(lo[1], 1e-9):,.0f}x fark"
        )
        print("  Esikler coin adedinden turedigi icin bu fark tasarim degil, artik.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
