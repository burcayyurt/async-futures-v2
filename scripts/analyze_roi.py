"""One-off: per-trade ROE analysis from the trade journal (truth source)."""
import json
from collections import defaultdict


def roe(r):
    return float(r["roe_pct"])


def show(label, rs):
    if not rs:
        return
    w = [r for r in rs if roe(r) >= 0]
    loss = [r for r in rs if roe(r) < 0]
    aw = sum(map(roe, w)) / max(len(w), 1)
    al = sum(map(roe, loss)) / max(len(loss), 1)
    print(f"\n### {label}: {len(rs)} trades, win {len(w) / len(rs) * 100:.0f}%, sumROE {sum(map(roe, rs)):+.1f}%")
    print(f"   wins {len(w):2} avg {aw:+.2f}%  | loss {len(loss):2} avg {al:+.2f}%")
    agg = defaultdict(lambda: [0, 0.0])
    for r in rs:
        agg[r["exit_reason"]][0] += 1
        agg[r["exit_reason"]][1] += roe(r)
    for k, (n, s) in sorted(agg.items()):
        print(f"     {k:14} n={n:2} sum={s:+7.1f}% avg={s / n:+6.2f}%")


def main():
    rows = [json.loads(line) for line in open("data/trades.jsonl") if line.strip()]
    pre = [r for r in rows if r["closed_at"][:10] == "2026-06-04"]
    post = [r for r in rows if r["closed_at"][:10] == "2026-06-05"]
    show("PRE-FIX (06-04)", pre)
    show("POST-FIX (06-05)", post)
    show("ALL", rows)

    src = post if post else rows
    w = [roe(r) for r in src if roe(r) >= 0]
    loss = [roe(r) for r in src if roe(r) < 0]
    aw = sum(w) / max(len(w), 1)
    al = abs(sum(loss) / max(len(loss), 1))
    be = al / (al + aw) * 100 if (al + aw) else 0
    print(f"\nPOST-FIX payoff: avgWin {aw:.2f}% vs avgLoss {al:.2f}% -> risk:reward 1:{al / max(aw, 1e-9):.1f}")
    print(f"Break-even win rate needed: {be:.0f}%  (actual {len(w) / len(src) * 100:.0f}%)")


if __name__ == "__main__":
    main()
