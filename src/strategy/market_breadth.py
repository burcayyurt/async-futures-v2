"""Portfolio-level directional breadth tracker (correlation-trap guard).

When many *distinct* symbols break out / cascade in the same direction inside a
short window, the move is market beta (correlation), not a per-symbol edge.
Both strategy engines feed their detected move directions here and gate signal
emission when the market is moving in lockstep — this is what stops the bot from
chasing the tail end of a synchronized dump/pump (the "correlation trap").

The tracker is engine-agnostic: callers record a normalized move ``direction``
(``"up"`` / ``"down"``) per coin, and query :meth:`is_synchronized` before
committing to a trade.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

UP = "up"
DOWN = "down"


@dataclass(slots=True)
class _Mark:
    coin: str
    direction: str
    ts: datetime


class MarketBreadthTracker:
    """Counts distinct same-direction moves across the watchlist in a window."""

    def __init__(self, window_seconds: int, max_same_side: int) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._max_same_side = max_same_side
        self._marks: deque[_Mark] = deque()

    def record(self, coin: str, direction: str, now: datetime) -> None:
        self._marks.append(_Mark(coin.strip().upper(), direction, now))
        self._prune(now)

    def _prune(self, now: datetime) -> None:
        cutoff = now - self._window
        while self._marks and self._marks[0].ts < cutoff:
            self._marks.popleft()

    def count_same_side(
        self, direction: str, now: datetime, *, exclude: str | None = None
    ) -> int:
        """Number of *distinct* coins moving ``direction`` within the window.

        ``exclude`` (the coin under evaluation) is never counted against itself.
        """
        self._prune(now)
        excluded = exclude.strip().upper() if exclude else None
        coins = {
            mark.coin
            for mark in self._marks
            if mark.direction == direction and mark.coin != excluded
        }
        return len(coins)

    def is_synchronized(
        self, direction: str, now: datetime, *, exclude: str | None = None
    ) -> bool:
        """True when enough *other* coins already moved the same direction.

        With ``max_same_side=4``, once four other symbols have moved ``direction``
        in the window the move is treated as correlated beta and the caller
        should skip it.
        """
        if self._max_same_side <= 0:
            return False
        return self.count_same_side(direction, now, exclude=exclude) >= self._max_same_side
