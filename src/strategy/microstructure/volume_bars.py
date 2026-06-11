"""Volume bars: sample the market by traded volume instead of by time.

Per the microstructure research, volume bars exhibit much lower serial
correlation than time bars and return distributions closer to Gaussian, which
makes z-scores and other statistical gates far more reliable. A bar closes once
the accumulated traded size reaches a per-symbol threshold; each bar also
carries the signed (buyer- minus seller-initiated) volume so downstream
order-flow features can be computed on bar boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.strategy.microstructure.flow_imbalance import trade_sign


@dataclass(slots=True)
class VolumeBar:
    """A completed (or in-progress) volume bar."""

    symbol: str
    open_px: Decimal
    high_px: Decimal
    low_px: Decimal
    close_px: Decimal
    volume: Decimal
    signed_volume: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    trade_count: int
    open_ts: datetime
    close_ts: datetime

    @property
    def imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        if total <= 0:
            return 0.0
        return float((self.buy_volume - self.sell_volume) / total)

    @property
    def ret(self) -> float:
        """Log-free simple return of the bar ``(close - open) / open``."""

        if self.open_px <= 0:
            return 0.0
        return float((self.close_px - self.open_px) / self.open_px)


@dataclass(slots=True)
class _Accumulator:
    open_px: Decimal
    high_px: Decimal
    low_px: Decimal
    close_px: Decimal
    volume: Decimal = Decimal("0")
    signed_volume: Decimal = Decimal("0")
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    trade_count: int = 0
    open_ts: datetime = field(default=None)  # type: ignore[assignment]
    close_ts: datetime = field(default=None)  # type: ignore[assignment]


class VolumeBarAggregator:
    """Aggregate trades into fixed-volume bars for a single symbol.

    Feed trades via :meth:`update`; whenever the accumulated volume crosses the
    ``threshold`` the method returns one (or more) completed :class:`VolumeBar`
    instances. Residual volume rolls into the next bar.
    """

    __slots__ = ("_symbol", "_threshold", "_acc")

    def __init__(self, symbol: str, threshold: Decimal | float) -> None:
        threshold_dec = (
            threshold if isinstance(threshold, Decimal) else Decimal(str(threshold))
        )
        if threshold_dec <= 0:
            raise ValueError("threshold must be > 0")
        self._symbol = symbol.strip().upper()
        self._threshold = threshold_dec
        self._acc: _Accumulator | None = None

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def threshold(self) -> Decimal:
        return self._threshold

    @property
    def has_partial(self) -> bool:
        return self._acc is not None

    def update(
        self,
        px: Decimal,
        size: Decimal,
        side: str,
        ts: datetime,
    ) -> list[VolumeBar]:
        """Add one trade. Returns the list of bars completed by this trade."""

        if size <= 0:
            return []

        sign = trade_sign(side)
        completed: list[VolumeBar] = []
        remaining = size

        while remaining > 0:
            if self._acc is None:
                self._acc = _Accumulator(
                    open_px=px,
                    high_px=px,
                    low_px=px,
                    close_px=px,
                    open_ts=ts,
                    close_ts=ts,
                )

            acc = self._acc
            capacity = self._threshold - acc.volume
            fill = remaining if remaining < capacity else capacity

            acc.high_px = max(acc.high_px, px)
            acc.low_px = min(acc.low_px, px)
            acc.close_px = px
            acc.close_ts = ts
            acc.volume += fill
            acc.trade_count += 1
            if sign > 0:
                acc.buy_volume += fill
                acc.signed_volume += fill
            elif sign < 0:
                acc.sell_volume += fill
                acc.signed_volume -= fill

            remaining -= fill

            if acc.volume >= self._threshold:
                completed.append(
                    VolumeBar(
                        symbol=self._symbol,
                        open_px=acc.open_px,
                        high_px=acc.high_px,
                        low_px=acc.low_px,
                        close_px=acc.close_px,
                        volume=acc.volume,
                        signed_volume=acc.signed_volume,
                        buy_volume=acc.buy_volume,
                        sell_volume=acc.sell_volume,
                        trade_count=acc.trade_count,
                        open_ts=acc.open_ts,
                        close_ts=acc.close_ts,
                    )
                )
                self._acc = None

        return completed
