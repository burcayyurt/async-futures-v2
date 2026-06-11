"""Parameter sweep for the DVSLA strategy over recorded events.

Loads a recording once into memory, then runs the backtest simulator for each
point in a parameter grid (cartesian product of supplied DVSLA + simulator
overrides) and ranks the results by a chosen metric. Use it to find robust
threshold regions — not a single overfit peak.

Example
-------
>>> from backtest.replay import EventReplayer
>>> from backtest.sweep import sweep, default_grid
>>> events = list(EventReplayer.from_directory("data/recordings"))
>>> results = sweep(events, default_grid())
>>> for r in results[:5]:
...     print(r.params, r.metrics.profit_factor)
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from backtest.metrics import Metrics, compute_metrics
from backtest.simulator import BacktestSimulator, SimConfig
from src.exchange.hyperliquid_ws import MarketEvent
from src.strategy.dvsla import DvslaParams, DvslaStrategy

# Fields routed to DvslaParams vs SimConfig so a flat grid can target both.
_DVSLA_FIELDS = frozenset(DvslaParams.model_fields)
_SIM_FIELDS = frozenset(
    f.name for f in SimConfig.__dataclass_fields__.values()  # type: ignore[attr-defined]
)


@dataclass(slots=True)
class SweepResult:
    params: dict[str, object]
    metrics: Metrics


def default_grid() -> dict[str, Sequence[object]]:
    """A sensible starting grid for the FAZ-0-Lite feed."""

    return {
        "ret_z_entry": [Decimal("2.5"), Decimal("3.0"), Decimal("3.5")],
        "oi_z_drop": [Decimal("-1.5"), Decimal("-2.0"), Decimal("-2.5")],
        "flow_imbalance_min": [Decimal("0.4"), Decimal("0.5"), Decimal("0.6")],
        "hurst_max": [Decimal("0.45"), Decimal("0.5")],
        "take_profit_pct": [Decimal("0.004"), Decimal("0.006"), Decimal("0.008")],
        "stop_loss_pct": [Decimal("0.004"), Decimal("0.006")],
    }


def _split_params(
    combo: dict[str, object],
    base_dvsla: dict[str, object],
    base_sim: dict[str, object],
) -> tuple[DvslaParams, SimConfig]:
    dvsla_kwargs = dict(base_dvsla)
    sim_kwargs = dict(base_sim)
    for key, value in combo.items():
        if key in _DVSLA_FIELDS:
            dvsla_kwargs[key] = value
        elif key in _SIM_FIELDS:
            sim_kwargs[key] = value
        else:
            raise KeyError(f"Unknown sweep parameter: {key!r}")
    return DvslaParams(**dvsla_kwargs), SimConfig(**sim_kwargs)


def _iter_combos(grid: dict[str, Sequence[object]]) -> Iterable[dict[str, object]]:
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def sweep(
    events: Sequence[MarketEvent],
    grid: dict[str, Sequence[object]] | None = None,
    *,
    base_dvsla: dict[str, object] | None = None,
    base_sim: dict[str, object] | None = None,
    rank_by: str = "profit_factor",
    min_trades: int = 5,
) -> list[SweepResult]:
    """Run the simulator across the grid; return results sorted best-first.

    Results with fewer than ``min_trades`` are kept but sorted last so a single
    lucky trade cannot top the ranking.
    """

    grid = grid or default_grid()
    base_dvsla = base_dvsla or {}
    base_sim = base_sim or {}
    event_list = list(events)
    results: list[SweepResult] = []

    for combo in _iter_combos(grid):
        params, sim_cfg = _split_params(combo, base_dvsla, base_sim)
        strategy = DvslaStrategy(params)
        sim = BacktestSimulator(strategy, sim_cfg)
        run = sim.run(event_list)
        metrics = compute_metrics(run.trades)
        results.append(SweepResult(params=combo, metrics=metrics))

    def _key(result: SweepResult) -> tuple[int, float]:
        enough = result.metrics.trades >= min_trades
        score = getattr(result.metrics, rank_by)
        score_f = float(score)
        if score_f == float("inf"):
            score_f = 1e9
        return (1 if enough else 0, score_f)

    results.sort(key=_key, reverse=True)
    return results
