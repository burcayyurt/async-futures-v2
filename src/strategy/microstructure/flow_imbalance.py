"""Order-flow imbalance from trade aggressor side (VOI proxy).

Hyperliquid does not expose L2 book volumes over the FAZ-0-Lite feed, so we
approximate the Volume Order Imbalance using the *aggressor* side reported on
each public trade. A taker buy adds to buy volume, a taker sell adds to sell
volume; the normalized difference is a strong directional micro-signal.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal


def trade_sign(side: str) -> int:
    """Map a Hyperliquid trade ``side`` to an aggressor sign.

    Hyperliquid reports the taker side as ``"B"`` (buy / bid-aggressor) or
    ``"A"`` (sell / ask-aggressor). Some payloads use ``"buy"``/``"sell"``.
    Returns ``+1`` for buyer-initiated, ``-1`` for seller-initiated, ``0`` when
    unknown.
    """

    normalized = side.strip().lower()
    if normalized in {"b", "buy", "bid"}:
        return 1
    if normalized in {"a", "sell", "ask"}:
        return -1
    return 0


def signed_volume(side: str, size: Decimal | float) -> Decimal:
    """Return ``+size`` for buyer-initiated and ``-size`` for seller-initiated."""

    sign = trade_sign(side)
    size_dec = size if isinstance(size, Decimal) else Decimal(str(size))
    return size_dec * Decimal(sign)


def flow_imbalance(buy_volume: float, sell_volume: float) -> float:
    """Normalized trade-flow imbalance ``(Vb - Va) / (Vb + Va)`` in [-1, 1].

    Returns ``0.0`` when there is no volume on either side.
    """

    total = buy_volume + sell_volume
    if total <= 0.0:
        return 0.0
    return (buy_volume - sell_volume) / total


class FlowImbalanceTracker:
    """Rolling order-flow imbalance over the last ``window`` trades.

    Accumulates buyer- and seller-initiated notional/size and exposes the
    normalized imbalance. Old trades beyond ``window`` are evicted so the
    metric reflects recent flow only.
    """

    __slots__ = ("_window", "_buys", "_sells", "_buy_sum", "_sell_sum")

    def __init__(self, window: int = 200) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        # Each entry is (buy_amount, sell_amount) for a single trade.
        self._buys: deque[float] = deque(maxlen=window)
        self._sells: deque[float] = deque(maxlen=window)
        self._buy_sum = 0.0
        self._sell_sum = 0.0

    @property
    def window(self) -> int:
        return self._window

    @property
    def count(self) -> int:
        return len(self._buys)

    @property
    def buy_volume(self) -> float:
        return self._buy_sum

    @property
    def sell_volume(self) -> float:
        return self._sell_sum

    def update(self, side: str, size: Decimal | float) -> None:
        sign = trade_sign(side)
        amount = float(size)
        buy_amt = amount if sign > 0 else 0.0
        sell_amt = amount if sign < 0 else 0.0

        if len(self._buys) == self._window:
            self._buy_sum -= self._buys[0]
            self._sell_sum -= self._sells[0]
        self._buys.append(buy_amt)
        self._sells.append(sell_amt)
        self._buy_sum += buy_amt
        self._sell_sum += sell_amt

    @property
    def imbalance(self) -> float:
        return flow_imbalance(self._buy_sum, self._sell_sum)
