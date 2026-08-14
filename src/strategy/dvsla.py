"""DVSLA — Dynamic Volume-Synchronized Liquidity Absorption.

A mean-reversion-after-liquidation-cascade engine. The thesis: when a
leveraged-position cascade forces price to spike sharply in one direction on
one-sided aggressor flow *while open interest collapses* (positions being
liquidated, not fresh money entering), price tends to snap back toward fair
value. DVSLA detects that footprint on volume bars and fades the move with a
maker limit entry, but only while the regime is statistically mean-reverting
(Hurst ``H < hurst_max``).

Signal footprint (cascade direction must be confirmed) for a cascade:

* **Return shock** — the just-closed volume bar's return has an extreme
  z-score against recent bars (``|z| >= ret_z_entry``).
* **One-sided flow** — rolling order-flow imbalance is dominated by the
  aggressor side that drove the move (``|imbalance| >= flow_imbalance_min``).
* **Regime gate** — Hurst exponent over recent bar *returns* (measured on the
  lead-up, excluding the shock bar) is mean-reverting (``H < hurst_max``).

**Open-interest collapse** (sharply negative z-score of OI change) is *not* a
hard gate — the Hyperliquid OI feed is too sparse/stale to reliably confirm
cascades at bar-close moments. Instead ``oi_z`` is folded into the signal's
confidence score (a strong collapse raises confidence).

When a down-cascade is confirmed DVSLA emits a **LONG** (fade); an up-cascade
emits a **SHORT**. A per-symbol cooldown prevents stacking entries on the same
cascade.

This engine is feed-compatible with the legacy strategy: it exposes
``on_market_event`` / ``consume`` and emits the shared :class:`TradeSignal`.
Execution-layer concerns (maker limit placement, Volume-ATR stop, VWAP-reversion
take-profit, confidence sizing) are handled downstream in PR5.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)
from src.strategy.microstructure import (
    FlowImbalanceTracker,
    OIZScoreTracker,
    RollingStats,
    VolumeBar,
    VolumeBarAggregator,
    classify_hurst,
    hurst_rs,
)
from src.strategy.market_breadth import DOWN, UP, MarketBreadthTracker
from src.strategy.microstructure.hurst import HurstRegime
from src.strategy.signals import SignalSide, TradeSignal

logger = logging.getLogger(__name__)


class CascadeDirection(str, Enum):
    """Direction of the detected liquidation cascade (the move being faded)."""

    UP = "up"
    DOWN = "down"


class DvslaParams(BaseModel):
    """Tunable thresholds for the DVSLA engine."""

    # Volume-bar sizing. ``volume_bar_threshold`` is the default per-bar traded
    # size; ``symbol_thresholds`` overrides it per coin (very different ADV).
    volume_bar_threshold: Decimal = Field(default=Decimal("50"), gt=Decimal("0"))
    symbol_thresholds: dict[str, Decimal] = Field(default_factory=dict)

    # Return-shock detection.
    ret_z_window: int = Field(default=50, gt=1)
    ret_z_entry: Decimal = Field(default=Decimal("3.0"), gt=Decimal("0"))
    # Minimum absolute bar return (fraction) for a bar to count as a shock. A
    # near-flat window collapses the return variance, which can drive the
    # z-score to absurd values; an extreme z alone is therefore not enough — the
    # move itself must be at least this large (default 0.15%).
    ret_min_abs_pct: Decimal = Field(default=Decimal("0.0015"), ge=Decimal("0"))
    # Hard cap on |ret_z| fed downstream (collapsed-variance windows otherwise
    # yield z-scores in the millions).
    ret_z_clamp: Decimal = Field(default=Decimal("25"), gt=Decimal("0"))
    # Reject entries whose |ret_z| is at/above this level: a z near the clamp
    # ceiling is a variance-collapse / post-reconnect burst artefact, not a real
    # cascade. 0 disables. Keep below ``ret_z_clamp``.
    ret_z_reject: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    # One-sided flow confirmation.
    flow_window: int = Field(default=200, gt=0)
    flow_imbalance_min: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"))

    # OI-collapse confirmation (z-score of OI change; cascade => sharply negative).
    oi_z_window: int = Field(default=50, gt=1)
    oi_z_drop: Decimal = Field(default=Decimal("-1.0"), lt=Decimal("0"))
    # In fade mode, veto signals when OI is strongly *expanding* (oi_z positive):
    # fresh directional money, not a liquidation cascade — fading it is a knife.
    fade_oi_z_veto: bool = Field(default=True)

    # Hurst regime gate (computed on bar returns; H < hurst_max => mean-reverting).
    hurst_window: int = Field(default=64, gt=16)
    hurst_max: Decimal = Field(default=Decimal("0.55"), gt=Decimal("0"), lt=Decimal("1"))

    # Bars required before the engine is allowed to fire, and the post-signal
    # cooldown (in completed bars) before the same symbol can fire again.
    warmup_bars: int = Field(default=20, ge=0)
    cooldown_bars: int = Field(default=10, ge=0)

    # Trade direction. False (default) fades the cascade (original mean-reversion
    # thesis); True trades WITH the cascade (momentum continuation). Recorded-data
    # backtests on the FAZ-0-Lite feed show the fade has negative edge while the
    # momentum side is net-positive, so this flag lets the engine be inverted
    # without touching the detection logic.
    invert: bool = Field(default=False)

    model_config = {"arbitrary_types_allowed": True}

    def threshold_for(self, symbol: str) -> Decimal:
        return self.symbol_thresholds.get(symbol.strip().upper(), self.volume_bar_threshold)

    @classmethod
    def from_settings(cls, settings: "Any") -> "DvslaParams":
        return cls(
            volume_bar_threshold=settings.dvsla_volume_bar_threshold,
            symbol_thresholds=dict(settings.dvsla_symbol_thresholds),
            ret_z_window=settings.dvsla_ret_z_window,
            ret_z_entry=settings.dvsla_ret_z_entry,
            ret_min_abs_pct=settings.dvsla_ret_min_abs_pct,
            ret_z_clamp=settings.dvsla_ret_z_clamp,
            ret_z_reject=settings.dvsla_ret_z_reject,
            flow_window=settings.dvsla_flow_window,
            flow_imbalance_min=settings.dvsla_flow_imbalance_min,
            oi_z_window=settings.dvsla_oi_z_window,
            oi_z_drop=settings.dvsla_oi_z_drop,
            fade_oi_z_veto=settings.dvsla_fade_oi_z_veto,
            hurst_window=settings.dvsla_hurst_window,
            hurst_max=settings.dvsla_hurst_max,
            warmup_bars=settings.dvsla_warmup_bars,
            cooldown_bars=settings.dvsla_cooldown_bars,
            invert=settings.dvsla_invert,
        )


@dataclass
class _SymbolState:
    aggregator: VolumeBarAggregator
    flow: FlowImbalanceTracker
    returns: RollingStats
    oi_z: OIZScoreTracker
    bar_rets: deque[float]
    bar_count: int = 0
    last_signal_bar: int | None = None
    mark_px: Decimal | None = None
    oracle_px: Decimal | None = None
    last_oi_z: float = 0.0


@dataclass(slots=True)
class CascadeSignal:
    """Diagnostic record describing why a signal fired (handy for tests/logs)."""

    direction: CascadeDirection
    ret_z: float
    imbalance: float
    oi_z: float
    hurst: float
    divergence_pct: float


class DvslaStrategy:
    """Liquidation-cascade mean-reversion engine."""

    def __init__(
        self,
        params: DvslaParams | None = None,
        on_signal: Callable[[TradeSignal], None] | None = None,
        breadth: MarketBreadthTracker | None = None,
    ) -> None:
        self.params = params or DvslaParams()
        self._on_signal = on_signal
        self._breadth = breadth
        self._symbols: dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------ feed

    async def consume(self, queue: asyncio.Queue[MarketEvent]) -> None:
        while True:
            event = await queue.get()
            try:
                await self.on_market_event(event)
            except Exception:
                logger.exception("DVSLA failed processing market event")

    async def on_market_event(self, event: MarketEvent) -> TradeSignal | None:
        if event.coin is None:
            return None

        coin = event.coin.strip().upper()
        now = event.ts if event.ts.tzinfo is not None else event.ts.replace(tzinfo=timezone.utc)
        state = self._state(coin)

        if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
            state.mark_px = event.payload.mark_px
            state.oracle_px = event.payload.oracle_px
            state.last_oi_z = state.oi_z.update(event.payload.open_interest)
            return None

        if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
            return self._on_trade(coin, state, event.payload, now)

        return None

    # ---------------------------------------------------------------- summary

    def decision_summary(self, watchlist: tuple[str, ...]) -> str:
        """Compact per-symbol engine state for periodic activity logging."""
        parts: list[str] = []
        for symbol in watchlist:
            coin = symbol.strip().upper()
            state = self._symbols.get(coin)
            if state is None:
                parts.append(f"{coin}:idle")
                continue
            if state.bar_count < self.params.warmup_bars:
                parts.append(f"{coin}:warmup({state.bar_count}/{self.params.warmup_bars})")
                continue
            parts.append(
                f"{coin}:bars={state.bar_count} imb={state.flow.imbalance:+.2f} "
                f"oi_z={state.last_oi_z:+.2f}"
            )
        return "engine=dvsla | " + " ".join(parts)

    # -------------------------------------------------------------- internals

    def _state(self, coin: str) -> _SymbolState:
        if coin not in self._symbols:
            self._symbols[coin] = _SymbolState(
                aggregator=VolumeBarAggregator(coin, self.params.threshold_for(coin)),
                flow=FlowImbalanceTracker(window=self.params.flow_window),
                returns=RollingStats(self.params.ret_z_window),
                oi_z=OIZScoreTracker(self.params.oi_z_window),
                bar_rets=deque(maxlen=self.params.hurst_window),
            )
        return self._symbols[coin]

    def _on_trade(
        self,
        coin: str,
        state: _SymbolState,
        payload: TradePayload,
        now: datetime,
    ) -> TradeSignal | None:
        state.flow.update(payload.side, payload.sz)
        if state.mark_px is None:
            state.mark_px = payload.px

        bars = state.aggregator.update(payload.px, payload.sz, payload.side, now)
        signal: TradeSignal | None = None
        for bar in bars:
            produced = self._on_bar_close(coin, state, bar, now)
            if produced is not None:
                signal = produced
        return signal

    def _on_bar_close(
        self,
        coin: str,
        state: _SymbolState,
        bar: VolumeBar,
        now: datetime,
    ) -> TradeSignal | None:
        ret = bar.ret
        # Score this bar's return against the window *before* incorporating it,
        # then record it — same discipline as the OI z-score tracker.
        ret_z = state.returns.z_score(ret) if state.returns.count >= 2 else 0.0
        # Clamp pathological z-scores: a near-flat window collapses the variance
        # and can drive |z| into the millions, making the entry gate meaningless.
        clamp = float(self.params.ret_z_clamp)
        ret_z = max(-clamp, min(clamp, ret_z))
        state.returns.update(ret)
        # Hurst is measured on bar *returns*, not price levels: R/S on an
        # integrated (price) series is persistent by construction (H~0.9) and
        # would never register the mean-reverting regime the strategy needs.
        state.bar_rets.append(float(ret))
        state.bar_count += 1

        if state.bar_count < self.params.warmup_bars:
            return None
        if not self._cooldown_elapsed(state):
            return None

        # A genuine cascade must be a real move, not just a statistical outlier
        # against a collapsed-variance window.
        if abs(ret) < float(self.params.ret_min_abs_pct):
            return None

        # Reject clamp-ceiling z-scores: these are variance-collapse or post-
        # reconnect burst artefacts rather than real cascades, and were the sole
        # large loser in the invert-era dry-run.
        reject = float(self.params.ret_z_reject)
        if reject > 0 and abs(ret_z) >= reject:
            logger.info(
                "DVSLA skip %s: ret_z=%.2f at/above reject threshold %.2f (artefact)",
                coin,
                ret_z,
                reject,
            )
            return None

        direction = self._classify_cascade(ret_z)
        if direction is None:
            return None
        if not self._flow_confirms(state, direction):
            return None

        # Fade guard: a strongly *positive* OI z-score means open interest is
        # expanding on the move (fresh directional money), not a liquidation
        # cascade. Fading that is catching a knife against a real trend, so veto
        # it. Only in fade mode — in momentum mode (invert) OI expansion actually
        # supports the continuation, so it is not a veto there.
        if (
            self.params.fade_oi_z_veto
            and not self.params.invert
            and state.last_oi_z >= abs(float(self.params.oi_z_drop))
        ):
            logger.info(
                "DVSLA skip %s: OI expanding (oi_z=%.2f) — trend, not cascade (fade veto)",
                coin,
                state.last_oi_z,
            )
            return None

        # Record the cascade's *move* direction for the correlation guard. A
        # cascade is detected here regardless of the eventual trade side (fade or
        # momentum), so breadth reflects the underlying market move.
        move = UP if direction is CascadeDirection.UP else DOWN
        if self._breadth is not None:
            self._breadth.record(coin, move, now)

        # OI collapse is no longer a hard veto: the Hyperliquid OI feed is too
        # sparse/stale to reliably confirm cascades at bar-close moments. It is
        # folded into the confidence score instead (see ``_confidence``).
        hurst, regime = self._regime(state)
        if regime is not HurstRegime.MEAN_REVERTING:
            return None

        # Correlation trap: if the broader market is cascading the same way, this
        # is beta, not a per-symbol edge — skip rather than trade the wave.
        if self._breadth is not None and self._breadth.is_synchronized(
            move, now, exclude=coin
        ):
            logger.info(
                "DVSLA skip %s: correlated %s move across market (breadth guard)",
                coin,
                move,
            )
            return None

        divergence_pct = self._divergence_pct(state)
        cascade = CascadeSignal(
            direction=direction,
            ret_z=ret_z,
            imbalance=state.flow.imbalance,
            oi_z=state.last_oi_z,
            hurst=hurst,
            divergence_pct=divergence_pct,
        )
        return self._emit(coin, state, bar, cascade, now)

    def _cooldown_elapsed(self, state: _SymbolState) -> bool:
        if state.last_signal_bar is None:
            return True
        return state.bar_count - state.last_signal_bar >= self.params.cooldown_bars

    def _classify_cascade(self, ret_z: float) -> CascadeDirection | None:
        entry = float(self.params.ret_z_entry)
        if ret_z <= -entry:
            return CascadeDirection.DOWN
        if ret_z >= entry:
            return CascadeDirection.UP
        return None

    def _flow_confirms(self, state: _SymbolState, direction: CascadeDirection) -> bool:
        imbalance = state.flow.imbalance
        threshold = float(self.params.flow_imbalance_min)
        if direction is CascadeDirection.DOWN:
            return imbalance <= -threshold
        return imbalance >= threshold

    def _regime(self, state: _SymbolState) -> tuple[float, HurstRegime]:
        # Assess the regime on the *lead-up* to the bar, excluding the bar just
        # appended: a large cascade return would otherwise inflate the Hurst
        # reading and mask the mean-reverting regime we want to detect.
        rets = list(state.bar_rets)[:-1]
        if len(rets) < 16:
            return 0.5, HurstRegime.RANDOM_WALK
        hurst = hurst_rs(rets)
        return hurst, classify_hurst(hurst, lower=float(self.params.hurst_max))

    @staticmethod
    def _divergence_pct(state: _SymbolState) -> float:
        if state.mark_px is None or state.oracle_px is None or state.oracle_px <= 0:
            return 0.0
        return float((state.mark_px - state.oracle_px) / state.oracle_px) * 100.0

    def _emit(
        self,
        coin: str,
        state: _SymbolState,
        bar: VolumeBar,
        cascade: CascadeSignal,
        now: datetime,
    ) -> TradeSignal:
        # Fade the cascade: a down-cascade is bought, an up-cascade is sold.
        # When ``invert`` is set, trade WITH the cascade instead (momentum): a
        # down-cascade is sold and an up-cascade is bought.
        fade = SignalSide.LONG if cascade.direction is CascadeDirection.DOWN else SignalSide.SHORT
        if self.params.invert:
            side = SignalSide.SHORT if fade is SignalSide.LONG else SignalSide.LONG
        else:
            side = fade
        entry_px = state.mark_px if state.mark_px is not None else bar.close_px
        confidence = self._confidence(cascade)
        state.last_signal_bar = state.bar_count

        signal = TradeSignal(
            symbol=coin,
            side=side,
            entry_mark_price=entry_px,
            confidence=confidence,
            timestamp=now,
            reason=(
                f"dvsla_{'momentum' if self.params.invert else 'fade'}_"
                f"{cascade.direction.value} "
                f"ret_z={cascade.ret_z:.2f} imb={cascade.imbalance:.2f} "
                f"oi_z={cascade.oi_z:.2f} H={cascade.hurst:.2f}"
            ),
        )
        logger.info(
            "DVSLA signal %s %s | entry=%s | ret_z=%.2f imb=%.2f oi_z=%.2f H=%.2f div=%.3f%% | conf=%.2f",
            coin,
            side.value.upper(),
            entry_px,
            cascade.ret_z,
            cascade.imbalance,
            cascade.oi_z,
            cascade.hurst,
            cascade.divergence_pct,
            confidence,
        )
        if self._on_signal is not None:
            self._on_signal(signal)
        return signal

    def _confidence(self, cascade: CascadeSignal) -> Decimal:
        ret_entry = float(self.params.ret_z_entry)
        oi_drop = abs(float(self.params.oi_z_drop))
        flow_min = float(self.params.flow_imbalance_min) or 1.0

        # Each component is the excess beyond its trigger, normalized to ~[0, 1].
        ret_score = min(1.0, abs(cascade.ret_z) / (ret_entry * 2.0)) if ret_entry > 0 else 0.0
        flow_score = min(1.0, abs(cascade.imbalance) / min(1.0, flow_min * 2.0))
        oi_score = min(1.0, abs(cascade.oi_z) / (oi_drop * 2.0)) if oi_drop > 0 else 0.0
        # Stronger mean-reversion (lower Hurst) => higher confidence.
        hurst_score = min(1.0, max(0.0, (0.5 - cascade.hurst) / 0.5))

        raw = (ret_score + flow_score + oi_score + hurst_score) / 4.0
        return Decimal(str(round(min(1.0, max(0.0, raw)), 4)))
