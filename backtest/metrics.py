"""Performance metrics for a backtest run.

Turns a list of :class:`~backtest.simulator.ClosedTrade` into the standard
quant scorecard: win rate, profit factor, expectancy, average R, max drawdown
and a per-trade Sharpe ratio. All money math stays in :class:`~decimal.Decimal`
to match the simulator; ratios are returned as floats for reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

from backtest.simulator import ClosedTrade


@dataclass(slots=True)
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    net_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    expectancy: Decimal = Decimal("0")
    avg_r: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    sharpe: float = 0.0
    total_fees: Decimal = Decimal("0")
    exit_breakdown: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "net_pnl": float(self.net_pnl),
            "gross_profit": float(self.gross_profit),
            "gross_loss": float(self.gross_loss),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": float(self.avg_win),
            "avg_loss": float(self.avg_loss),
            "expectancy": float(self.expectancy),
            "avg_r": round(self.avg_r, 4),
            "max_drawdown": float(self.max_drawdown),
            "sharpe": round(self.sharpe, 4),
            "total_fees": float(self.total_fees),
            "exit_breakdown": dict(self.exit_breakdown),
        }


def _max_drawdown(equity_curve: list[Decimal]) -> Decimal:
    peak = Decimal("0")
    max_dd = Decimal("0")
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _sharpe(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(n)


def compute_metrics(trades: list[ClosedTrade]) -> Metrics:
    """Aggregate a list of closed trades into a :class:`Metrics` scorecard."""

    metrics = Metrics(trades=len(trades))
    if not trades:
        return metrics

    equity = Decimal("0")
    equity_curve: list[Decimal] = []
    r_values: list[float] = []
    pnl_returns: list[float] = []

    for trade in trades:
        net = trade.net_pnl
        metrics.net_pnl += net
        metrics.total_fees += trade.fees
        metrics.exit_breakdown[trade.exit_reason] = (
            metrics.exit_breakdown.get(trade.exit_reason, 0) + 1
        )

        if net > 0:
            metrics.wins += 1
            metrics.gross_profit += net
        elif net < 0:
            metrics.losses += 1
            metrics.gross_loss += -net

        equity += net
        equity_curve.append(equity)
        r_values.append(float(trade.r_multiple))
        # Per-trade return on the position notional (unitless).
        if trade.notional > 0:
            pnl_returns.append(float(net / trade.notional))

    metrics.win_rate = metrics.wins / metrics.trades
    if metrics.gross_loss > 0:
        metrics.profit_factor = float(metrics.gross_profit / metrics.gross_loss)
    elif metrics.gross_profit > 0:
        metrics.profit_factor = float("inf")

    metrics.avg_win = (
        metrics.gross_profit / metrics.wins if metrics.wins else Decimal("0")
    )
    metrics.avg_loss = (
        metrics.gross_loss / metrics.losses if metrics.losses else Decimal("0")
    )
    metrics.expectancy = metrics.net_pnl / metrics.trades
    metrics.avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    metrics.max_drawdown = _max_drawdown(equity_curve)
    metrics.sharpe = _sharpe(pnl_returns)

    return metrics
