"""Backfill fee accounting into an existing trade journal.

Records written before fee accounting existed carry gross PnL only, rounded to
whole cents. Both problems are recoverable: ``entry_px``, ``exit_px`` and
``size`` are stored at full precision, so gross PnL is recomputed exactly rather
than inherited from the rounded field.

Usage::

    python -m scripts.backfill_trade_fees                # dry run, prints summary
    python -m scripts.backfill_trade_fees --write        # rewrite the journal

A timestamped ``.bak`` copy is made before the file is rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.core.config import HyperliquidSettings
from src.core.trade_journal import calc_fee_usd


def gross_pnl(side: str, entry_px: Decimal, exit_px: Decimal, size: Decimal) -> Decimal:
    if side == "long":
        return (exit_px - entry_px) * size
    return (entry_px - exit_px) * size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the journal in place")
    parser.add_argument("--path", default=None, help="journal path (defaults to settings)")
    args = parser.parse_args()

    settings = HyperliquidSettings()
    path = Path(args.path or settings.trade_journal_path)
    if not path.exists():
        print(f"journal not found: {path}")
        return 1

    maker_entry = settings.maker_entry_enabled
    maker_bps = settings.maker_fee_bps
    taker_bps = settings.taker_fee_bps

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[str] = []
    tot_gross = tot_fee = tot_notional = Decimal("0")
    wins_gross = wins_net = 0

    for line in lines:
        rec = json.loads(line)
        entry_px = Decimal(str(rec["entry_px"]))
        exit_px = Decimal(str(rec["exit_px"]))
        size = Decimal(str(rec["size"]))
        side = str(rec["side"]).lower()

        g = gross_pnl(side, entry_px, exit_px, size)
        fee = calc_fee_usd(
            entry_px=entry_px,
            exit_px=exit_px,
            size=size,
            maker_entry=maker_entry,
            maker_fee_bps=maker_bps,
            taker_fee_bps=taker_bps,
        )
        net = g - fee

        rec["pnl_usd"] = str(g.quantize(Decimal("0.0001")))
        rec["fee_usd"] = str(fee.quantize(Decimal("0.0001")))
        rec["net_pnl_usd"] = str(net.quantize(Decimal("0.0001")))
        out.append(json.dumps(rec, ensure_ascii=False))

        tot_gross += g
        tot_fee += fee
        tot_notional += entry_px * size
        wins_gross += 1 if g > 0 else 0
        wins_net += 1 if net > 0 else 0

    n = len(out)
    entry_label = "maker" if maker_entry else "taker"
    print(f"journal        : {path}")
    print(f"trades         : {n}")
    print(f"fee model      : entry={entry_label} {maker_bps if maker_entry else taker_bps}bps, exit=taker {taker_bps}bps")
    print(f"total notional : ${tot_notional:,.2f}")
    print(f"gross PnL      : {tot_gross:+.2f}")
    print(f"fees           : -{tot_fee:.2f}")
    print(f"NET PnL        : {tot_gross - tot_fee:+.2f}")
    if n:
        print(f"win rate gross : {wins_gross / n * 100:.1f}%  ({wins_gross}/{n})")
        print(f"win rate net   : {wins_net / n * 100:.1f}%  ({wins_net}/{n})")

    if not args.write:
        print("\n(dry run — pass --write to rewrite the journal)")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, backup)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nbackup written : {backup}")
    print(f"journal rewritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
