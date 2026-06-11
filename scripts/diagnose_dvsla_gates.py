"""Diagnose which DVSLA gate blocks signals on recorded data.

Counts, per closed bar (after warmup/cooldown), how many bars pass each gate
in the 4-way conjunction: cascade (ret_z) -> flow -> OI -> Hurst regime.

Usage: python scripts/diagnose_dvsla_gates.py [recordings_dir]
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

from backtest.replay import EventReplayer
from src.core.config import HyperliquidSettings
from src.strategy.dvsla import (
    CascadeDirection,
    DvslaParams,
    DvslaStrategy,
    HurstRegime,
)

C: Counter = Counter()
DIST = {"ret_z_abs_max": 0.0, "oi_z_min": 0.0, "imb_abs_max": 0.0, "hurst_min": 1.0}
HURST_VALS: list[float] = []
OI_FLOW_VALS: list[float] = []  # oi_z at bars that passed cascade+flow


class DiagStrategy(DvslaStrategy):
    def _on_bar_close(self, coin, state, bar, now):  # type: ignore[override]
        ret = bar.ret
        ret_z = state.returns.z_score(ret) if state.returns.count >= 2 else 0.0
        # Mirror parent's bookkeeping by delegating, but instrument first.
        bar_count_after = state.bar_count + 1
        if bar_count_after >= self.params.warmup_bars and self._cooldown_elapsed(state):
            C["evaluated"] += 1
            DIST["ret_z_abs_max"] = max(DIST["ret_z_abs_max"], abs(ret_z))
            DIST["imb_abs_max"] = max(DIST["imb_abs_max"], abs(state.flow.imbalance))
            direction = self._classify_cascade(ret_z)
            if direction is not None:
                C["pass_cascade"] += 1
                if self._flow_confirms(state, direction):
                    C["pass_flow"] += 1
                    DIST["oi_z_min"] = min(DIST["oi_z_min"], state.last_oi_z)
                    if state.oi_z.is_ready:
                        OI_FLOW_VALS.append(state.last_oi_z)
                    # OI is now a confidence weight, not a gate. The regime gate
                    # is measured on the pre-shock window (bar_rets not yet
                    # holding this bar, since super() appends afterwards).
                    _, regime = self._regime(state)
                    if regime is HurstRegime.MEAN_REVERTING:
                        C["pass_hurst"] += 1
            if state.oi_z.is_ready:
                DIST["oi_z_min"] = min(DIST["oi_z_min"], state.last_oi_z)
                if len(state.bar_rets) >= 16:
                    h, _ = self._regime(state)
                    DIST["hurst_min"] = min(DIST["hurst_min"], h)
                    HURST_VALS.append(h)
        return super()._on_bar_close(coin, state, bar, now)


async def _run(directory: str):
    settings = HyperliquidSettings.from_env()
    params = DvslaParams.from_settings(settings)
    signals: list = []
    strat = DiagStrategy(params, on_signal=signals.append)
    n = 0
    bars = 0
    for event in EventReplayer.from_directory(directory):
        n += 1
        await strat.on_market_event(event)
    for st in strat._symbols.values():  # type: ignore[attr-defined]
        bars += st.bar_count
    return n, bars, signals


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/recordings"
    n, bars, signals = asyncio.run(_run(directory))
    print(f"Events: {n:,}  | total bars closed: {bars:,}  | signals: {len(signals)}")
    print("Gate funnel (count of bars passing each cumulative gate):")
    for k in ("evaluated", "pass_cascade", "pass_flow", "pass_hurst"):
        print(f"  {k:14} {C.get(k, 0):,}")
    print("Observed extremes across evaluated bars:")
    print(f"  max |ret_z|   = {DIST['ret_z_abs_max']:.2f}")
    print(f"  max |imb|     = {DIST['imb_abs_max']:.2f}")
    print(f"  min oi_z      = {DIST['oi_z_min']:.2f}")
    print(f"  min hurst     = {DIST['hurst_min']:.3f}")

    def pct(vals, ps):
        if not vals:
            return "(none)"
        s = sorted(vals)
        out = []
        for p in ps:
            idx = min(len(s) - 1, int(p / 100 * len(s)))
            out.append(f"p{p}={s[idx]:.3f}")
        return "  ".join(out)

    print(f"Hurst dist ({len(HURST_VALS)} samples): " + pct(HURST_VALS, [1, 5, 25, 50]))
    print(f"OI z at cascade+flow bars ({len(OI_FLOW_VALS)} samples): " + pct(OI_FLOW_VALS, [1, 5, 10, 25, 50]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
