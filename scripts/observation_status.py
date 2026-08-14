"""Where the dry-run observation stands: coverage, and results per configuration.

Answers the three questions that matter while evidence accumulates: was the bot
actually running, how many trades has the current configuration produced, and is
that yet enough to distinguish its edge from noise.

Trades are grouped by ``config_id`` so a parameter change starts a fresh, clean
sample instead of contaminating the previous one. Rows written before
fingerprinting existed are grouped as ``legacy`` and reported separately, since
their parameters cannot be reconstructed.

Usage:
    python -m scripts.observation_status
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import HyperliquidSettings
from src.core.session_log import coverage_pct, load_sessions

SIGNIFICANCE_T = 2.0
INTERESTING_FIELDS = (
    "trailing_callback_pct",
    "break_even_trigger_pct",
    "maker_entry_enabled",
    "atr_stop_min_pct",
    "dvsla_invert",
    "leverage",
)


def _load_registry(path: Path) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    if not path.exists():
        return registry
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        registry[entry.get("config_id", "?")] = entry
    return registry


def _trades_needed(mean: float, sd: float) -> int | None:
    """Sample size at which |t| would reach the significance bar."""
    if mean == 0 or sd <= 0:
        return None
    return math.ceil((SIGNIFICANCE_T * sd / abs(mean)) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=None)
    args = ap.parse_args()

    settings = HyperliquidSettings.from_env()
    journal_path = Path(args.journal or settings.trade_journal_path)
    registry = _load_registry(Path(settings.config_registry_path))
    sessions = load_sessions(settings.session_log_path, settings.session_state_path)

    print("=" * 78)
    print("OBSERVATION STATUS")
    print("=" * 78)

    if sessions:
        first, last = sessions[0].started_at, max(s.last_seen for s in sessions)
        span_h = (last - first).total_seconds() / 3600
        running = sum(s.duration_seconds for s in sessions) / 3600
        unclean = sum(1 for s in sessions if not s.clean_exit and not s.in_progress)
        live = sum(1 for s in sessions if s.in_progress)
        detail = f"{unclean} ended without clean shutdown"
        if live:
            detail += ", 1 running now"
        print(f"  sessions      : {len(sessions)}  ({detail})")
        print(f"  window        : {first:%Y-%m-%d %H:%M} -> {last:%Y-%m-%d %H:%M} UTC ({span_h:.1f}h)")
        print(f"  running time  : {running:.1f}h")
        print(f"  coverage      : {coverage_pct(sessions):.1f}%")
    else:
        print("  sessions      : none recorded yet (session tracking starts on next boot)")

    if not journal_path.exists():
        print(f"\nNo journal at {journal_path}")
        return 1

    rows: dict[str, list[dict]] = {}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.setdefault(str(rec.get("config_id", "legacy")), []).append(rec)

    print("\n" + "-" * 78)
    print("RESULTS BY CONFIGURATION  (net of fees)")
    print("-" * 78)
    print(f"{'config_id':>14} {'n':>5} {'win%':>5} {'net$':>9} {'mean$':>9} {'t':>7}  progress")
    print("-" * 78)

    for cid, recs in sorted(rows.items(), key=lambda kv: len(kv[1])):
        nets = [float(r.get("net_pnl_usd", r.get("pnl_usd", 0))) for r in recs]
        n = len(nets)
        total = sum(nets)
        wins = sum(1 for x in nets if x > 0)
        if n < 2:
            print(f"{cid:>14} {n:5d} {'':>5} {total:9.2f} {'':>9} {'':>7}  too few trades")
            continue
        mean = statistics.mean(nets)
        sd = statistics.stdev(nets)
        t = mean / (sd / math.sqrt(n)) if sd > 0 else float("nan")
        need = _trades_needed(mean, sd)
        if cid == "legacy":
            # These rows predate fingerprinting and span several parameter sets
            # (the engine was inverted mid-way), so their aggregate is a mix of
            # different strategies rather than a baseline anything can be
            # compared against.
            note = "mixed configurations - not a baseline"
        elif abs(t) >= SIGNIFICANCE_T:
            note = "SIGNIFICANT " + ("(profitable)" if mean > 0 else "(losing)")
        elif need is None:
            note = "-"
        elif need > n:
            note = f"{need - n} more trades for |t|>={SIGNIFICANCE_T:.0f}"
        else:
            note = "borderline"
        print(f"{cid:>14} {n:5d} {wins / n * 100:5.0f} {total:9.2f} {mean:9.4f} {t:+7.2f}  {note}")

    print("\n" + "-" * 78)
    print("CONFIGURATIONS")
    print("-" * 78)
    for cid in rows:
        entry = registry.get(cid)
        if entry is None:
            print(f"  {cid}: not in registry (pre-fingerprint trades)")
            continue
        s = entry.get("settings", {})
        bits = " ".join(f"{f}={s.get(f)}" for f in INTERESTING_FIELDS if f in s)
        print(f"  {cid}  first seen {entry.get('first_seen', '?')[:19]}")
        print(f"    {bits}")
        print(f"    symbols={len(s.get('symbols', []))}")

    now = datetime.now(timezone.utc)
    print(f"\nreport generated {now:%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
