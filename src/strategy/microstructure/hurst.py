"""Hurst exponent for regime classification via rescaled-range (R/S) analysis.

The Hurst exponent ``H`` characterizes the long-memory behaviour of a series:

* ``H < 0.5`` → mean-reverting (anti-persistent) — DVSLA is active here.
* ``H ≈ 0.5`` → random walk (no edge).
* ``H > 0.5`` → trending (persistent) — DVSLA stands down.

We use classic rescaled-range analysis over multiple lags and fit a line to
``log(R/S)`` vs ``log(lag)`` in log space; the slope estimates ``H``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


class HurstRegime(str, Enum):
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"
    TRENDING = "trending"


def hurst_rs(series: Sequence[float], min_lag: int = 2, max_lag: int | None = None) -> float:
    """Estimate the Hurst exponent of ``series`` using R/S analysis.

    Returns ``0.5`` (random walk) when the series is too short or degenerate to
    produce a stable estimate.
    """

    data = np.asarray(series, dtype=np.float64)
    n = data.size
    if n < 16:
        return 0.5
    if max_lag is None:
        max_lag = n // 2
    max_lag = max(min_lag + 1, min(max_lag, n - 1))

    lags = np.arange(min_lag, max_lag + 1)
    rs_values: list[float] = []
    valid_lags: list[int] = []

    for lag in lags:
        # Split the series into non-overlapping chunks of length `lag`.
        n_chunks = n // lag
        if n_chunks < 1:
            continue
        chunk_rs: list[float] = []
        for i in range(n_chunks):
            chunk = data[i * lag : (i + 1) * lag]
            mean = chunk.mean()
            deviations = chunk - mean
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = chunk.std(ddof=0)
            if s > 0:
                chunk_rs.append(r / s)
        if chunk_rs:
            rs_values.append(float(np.mean(chunk_rs)))
            valid_lags.append(int(lag))

    if len(rs_values) < 2:
        return 0.5

    log_lags = np.log(np.asarray(valid_lags, dtype=np.float64))
    log_rs = np.log(np.asarray(rs_values, dtype=np.float64))
    # Slope of the best-fit line is the Hurst exponent.
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    h = float(slope)
    # Clamp to a sane [0, 1] range; estimation noise can push slightly outside.
    return min(1.0, max(0.0, h))


def classify_hurst(
    hurst: float,
    lower: float = 0.45,
    upper: float = 0.55,
) -> HurstRegime:
    """Bucket a Hurst value into a regime using a neutral band ``[lower, upper]``."""

    if hurst < lower:
        return HurstRegime.MEAN_REVERTING
    if hurst > upper:
        return HurstRegime.TRENDING
    return HurstRegime.RANDOM_WALK
