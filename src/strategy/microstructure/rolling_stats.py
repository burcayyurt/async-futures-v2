"""Online rolling statistics: Welford mean/variance and z-score helpers.

These primitives keep numerically stable running estimates of the mean and
variance over a sliding window without storing the whole history, plus a
fixed-window variant backed by a deque for cases where exact windowed
statistics are required.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable


def z_score(value: float, mean: float, std: float) -> float:
    """Return the z-score of ``value`` given a ``mean`` and ``std``.

    Returns ``0.0`` when ``std`` is non-positive (degenerate distribution),
    which avoids division-by-zero blow-ups in signal gating.
    """

    if std <= 0.0 or math.isnan(std):
        return 0.0
    return (value - mean) / std


class Welford:
    """Numerically stable online mean/variance (Welford's algorithm).

    Tracks the running mean and (sample) variance incrementally. Suitable for
    unbounded streams where storing all observations is undesirable.
    """

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance (ddof=1). Returns 0.0 with fewer than 2 samples."""

        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def update(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(value)

    def z_score(self, value: float) -> float:
        return z_score(value, self._mean, self.std)


class RollingStats:
    """Exact fixed-window mean/variance over the most recent ``window`` values.

    Unlike :class:`Welford`, this discards observations older than ``window``
    so the statistics reflect only the current window. Backed by a deque; O(n)
    recompute is avoided by maintaining running sums.
    """

    __slots__ = ("_window", "_values", "_sum", "_sum_sq")

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._values: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    @property
    def window(self) -> int:
        return self._window

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def is_full(self) -> bool:
        return len(self._values) == self._window

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return self._sum / len(self._values)

    @property
    def variance(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        mean = self._sum / n
        var = (self._sum_sq / n) - (mean * mean)
        # Convert population variance to sample variance and clamp tiny
        # negative results from floating point error.
        var = max(var, 0.0) * n / (n - 1)
        return var

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def update(self, value: float) -> None:
        if len(self._values) == self._window:
            evicted = self._values[0]
            self._sum -= evicted
            self._sum_sq -= evicted * evicted
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(value)

    def z_score(self, value: float) -> float:
        return z_score(value, self.mean, self.std)
