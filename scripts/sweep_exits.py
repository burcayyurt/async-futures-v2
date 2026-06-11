"""Sweep the live exit model (initial stop -> break-even -> trailing) on recorded
DVSLA signals, to answer: "should we let positions breathe before locking?"

Unlike ``sweep_tp_sl.py`` (fixed TP/SL bracket), this replicates the *actual*
``PositionManager`` exit sequence so the parameters map 1:1 to ``.env``:

* initial hard stop at ``init_stop_pct`` below/above entry,
* break-even: once price has moved ``break_even_pct`` in favour, the stop jumps
  to entry +/- a fee buffer (locks ~breakeven),
* trailing: exit when price retraces ``trailing_pct`` from the best price.

Direction comes from the strategy as configured in ``.env`` (DVSLA_INVERT). The
grid is run in a single event pass: the strategy runs once, every signal is
fanned out to N independent exit simulations.

Usage: python scripts/sweep_exits.py [recordings_dir] [--days N] [--first N]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

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

LEVERAGE = 10.0
FEE_BUFFER = 0.001            # matches FEE_BUFFER_PCT (break-even lock buffer)
ROUNDTRIP_COST_PCT = 0.0008   # maker in + taker out + slippage ~ 8 bps (price)

# Grid (all as fractions of entry price).
BE_TRIGGERS = [0.001, 0.002, 0.003, 0.005, 0.010, 9.99]   # 9.99 = break-even off
TRAILING = [0.004, 0.008, 0.015, 0.025]
INIT_STOPS = [0.004, 0.006, 0.008]

CURRENT = (0.001, 0.004, 0.004)  # .env baseline: BE +0.1%, trail 0.4%, stop 0.4%


@dataclass(slots=True)
class _Pos:
    side: SignalSide
    entry: float
    stop: float
    peak: float
    be_armed: bool = False


@dataclass(slots=True)
class _Cfg:
    be: float
    trail: float
    init_stop: float
    open: dict[str, _Pos] = None  # type: ignore[assignment]
    rets: list[tuple[float, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.open = {}
        self.rets = []


def _price_and_ts(event: MarketEvent):
    if event.coin is None:
        return None
    if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
        return float(event.payload.px)
    if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
        return float(event.payload.mark_px)
    return None


def _signed(side: SignalSide, entry: float, px: float) -> float:
    r = (px - entry) / entry
    return r if side == SignalSide.LONG else -r


def _step(cfg: _Cfg, coin: str, px: float) -> None:
    pos = cfg.open.get(coin)
    if pos is None:
        return
    # 1) peak
    if pos.side == SignalSide.LONG:
        pos.peak = max(pos.peak, px)
    else:
        pos.peak = min(pos.peak, px)
    # 2) break-even
    if not pos.be_armed and _signed(pos.side, pos.entry, px) >= cfg.be:
        if pos.side == SignalSide.LONG:
            pos.stop = max(pos.stop, pos.entry * (1 + FEE_BUFFER))
        else:
            pos.stop = min(pos.stop, pos.entry * (1 - FEE_BUFFER))
        pos.be_armed = True
    # 3) hard stop
    hit = (pos.side == SignalSide.LONG and px <= pos.stop) or (
        pos.side == SignalSide.SHORT and px >= pos.stop
    )
    if hit:
        _close(cfg, coin, px, "stop")
        return
    # 4) trailing
    if pos.side == SignalSide.LONG:
        trig = pos.peak * (1 - cfg.trail)
        if px <= trig:
            _close(cfg, coin, px, "trail")
    else:
        trig = pos.peak * (1 + cfg.trail)
        if px >= trig:
            _close(cfg, coin, px, "trail")


def _close(cfg: _Cfg, coin: str, px: float, reason: str) -> None:
    pos = cfg.open.pop(coin)
    gross = _signed(pos.side, pos.entry, px)
    roe = (gross - ROUNDTRIP_COST_PCT) * LEVERAGE * 100.0
    # stop in profit (after break-even) is really a break-even/lock exit
    if reason == "stop" and pos.be_armed and roe > 0:
        reason = "be_lock"
    cfg.rets.append((roe, reason))


def _open(cfg: _Cfg, coin: str, side: SignalSide, entry: float, init_stop: float) -> None:
    if coin in cfg.open:
        return
    stop = entry * (1 - init_stop) if side == SignalSide.LONG else entry * (1 + init_stop)
    cfg.open[coin] = _Pos(side=side, entry=entry, stop=stop, peak=entry)


def _report(cfg: _Cfg) -> dict:
    rets = [r for r, _ in cfg.rets]
    n = len(rets)
    if n == 0:
        return {"n": 0}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gp = sum(wins)
    gl = -sum(losses)
    from collections import Counter
    reasons = Counter(reason for _, reason in cfg.rets)
    return {
        "n": n,
        "win": len(wins) / n * 100,
        "avgW": gp / len(wins) if wins else 0.0,
        "avgL": -gl / len(losses) if losses else 0.0,
        "sum": sum(rets),
        "pf": gp / gl if gl > 0 else float("inf"),
        "exp": sum(rets) / n,
        "reasons": dict(reasons),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--first", type=int, default=None)
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    strategy = DvslaStrategy(params)
    print(f"DVSLA invert (momentum) = {params.invert}")

    cfgs = [_Cfg(be, tr, st) for be, tr, st in itertools.product(BE_TRIGGERS, TRAILING, INIT_STOPS)]

    files = sorted(Path(args.directory).glob("events-*.jsonl"))
    if args.first:
        files = files[: args.first]
    elif args.days:
        files = files[-args.days :]
    print(f"Files: {[f.name for f in files]}  | grid points: {len(cfgs)}")

    async def run() -> int:
        n = 0
        for path in files:
            for event in replay_file(path):
                n += 1
                px = _price_and_ts(event)
                if px is not None and event.coin is not None and px > 0:
                    coin = event.coin.strip().upper()
                    for cfg in cfgs:
                        _step(cfg, coin, px)
                signal = await strategy.on_market_event(event)
                if signal is not None:
                    coin = signal.symbol.strip().upper()
                    entry = float(signal.entry_mark_price)
                    if entry > 0:
                        for cfg in cfgs:
                            _open(cfg, coin, signal.side, entry, cfg.init_stop)
        return n

    n = asyncio.run(run())
    # close remaining at last price seen per coin (mark-to-market)
    print(f"Replayed {n:,} events\n")

    rows = [(cfg, _report(cfg)) for cfg in cfgs]
    rows = [(c, r) for c, r in rows if r.get("n", 0) >= 5]
    rows.sort(key=lambda cr: (cr[1]["exp"]), reverse=True)

    print(f"{'BE%':>5} {'trail%':>7} {'stop%':>6} {'n':>4} {'win%':>5} "
          f"{'avgW':>6} {'avgL':>7} {'sumROE':>8} {'PF':>6} {'expROE':>7}  flag")
    print("-" * 80)
    for cfg, r in rows[:30]:
        flag = "<< CURRENT" if (cfg.be, cfg.trail, cfg.init_stop) == CURRENT else ""
        pf = r["pf"] if r["pf"] != float("inf") else 999.0
        print(f"{cfg.be * 100:5.1f} {cfg.trail * 100:7.1f} {cfg.init_stop * 100:6.1f} "
              f"{r['n']:4d} {r['win']:5.0f} {r['avgW']:+6.1f} {r['avgL']:+7.1f} "
              f"{r['sum']:+8.1f} {pf:6.2f} {r['exp']:+7.2f}  {flag}")

    # Always show the current baseline explicitly.
    print("\nCurrent .env baseline (BE 0.1% / trail 0.4% / stop 0.4%):")
    for cfg, r in rows:
        if (cfg.be, cfg.trail, cfg.init_stop) == CURRENT:
            print(f"  n={r['n']} win={r['win']:.0f}% avgW={r['avgW']:+.1f} avgL={r['avgL']:+.1f} "
                  f"sumROE={r['sum']:+.1f} PF={r['pf']:.2f} exp={r['exp']:+.2f}")
            print(f"  exit reasons: {r['reasons']}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
