"""Open-Interest normalization: the "fresh money" filter.

Absolute OI is meaningless on its own. What matters for the DVSLA thesis is how
*unusual* the current change in OI is relative to its recent behaviour. We track
the rolling z-score of OI changes so the strategy can distinguish:

* OI **rising** with price → fresh capital entering (trend / continuation).
* OI **falling** sharply during a price spike → positions being liquidated
  (cascade), which is the mean-reversion setup DVSLA targets.
"""

from __future__ import annotations

from decimal import Decimal

from src.strategy.microstructure.rolling_stats import RollingStats


class OIZScoreTracker:
    """Track the rolling z-score of open-interest changes.

    Feed successive OI observations via :meth:`update`; the tracker computes the
    delta from the previous observation and returns its z-score against the
    rolling window of recent deltas.

    Two robustness guards address sparse/stale OI feeds (common for lower-volume
    coins on Hyperliquid, whose ``activeAssetCtx`` repeats the same OI value
    ~90% of the time):

    * ``skip_unchanged`` — identical consecutive OI readings carry no new
      information and are *not* fed into the window. Polluting the window with
      zero-deltas collapses its variance, so a later genuine move scores an
      absurd z (we observed > 1e6). When OI is unchanged the last meaningful
      z-score is carried forward instead.
    * ``z_clamp`` — the returned z-score is bounded to ``[-z_clamp, z_clamp]``
      as a final safety net against degenerate windows.
    """

    __slots__ = ("_stats", "_prev_oi", "_last_delta", "_last_z", "_skip_unchanged", "_z_clamp")

    def __init__(
        self,
        window: int = 50,
        *,
        skip_unchanged: bool = True,
        z_clamp: float = 10.0,
    ) -> None:
        self._stats = RollingStats(window)
        self._prev_oi: Decimal | None = None
        self._last_delta = 0.0
        self._last_z = 0.0
        self._skip_unchanged = skip_unchanged
        self._z_clamp = abs(z_clamp)

    @property
    def window(self) -> int:
        return self._stats.window

    @property
    def count(self) -> int:
        return self._stats.count

    @property
    def is_ready(self) -> bool:
        return self._stats.count >= 2

    @property
    def last_delta(self) -> float:
        return self._last_delta

    @property
    def last_z(self) -> float:
        return self._last_z

    def update(self, open_interest: Decimal) -> float:
        """Record a new OI observation; return the z-score of its change.

        Returns ``0.0`` on the first observation (no prior delta) and until the
        window holds at least two deltas. Identical consecutive readings (zero
        delta) carry the previous z-score forward unchanged when
        ``skip_unchanged`` is set.
        """

        if self._prev_oi is None:
            self._prev_oi = open_interest
            self._last_delta = 0.0
            self._last_z = 0.0
            return 0.0

        delta = float(open_interest - self._prev_oi)
        self._prev_oi = open_interest

        # A repeated OI value is a stale tick, not a real change. Feeding the
        # zero into the window would deflate its variance and blow up the next
        # real delta's z-score, so carry the last meaningful reading forward.
        if delta == 0.0 and self._skip_unchanged:
            return self._last_z

        self._last_delta = delta

        # z-score uses the window *before* adding the new delta so the current
        # observation is scored against its history, then we incorporate it.
        z = self._stats.z_score(delta) if self.is_ready else 0.0
        self._stats.update(delta)
        z = max(-self._z_clamp, min(self._z_clamp, z))
        self._last_z = z
        return z
