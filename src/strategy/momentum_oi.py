from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, MarketEvent, TradePayload
from src.strategy.market_breadth import DOWN, UP, MarketBreadthTracker
from src.strategy.signals import SignalSide, TradeSignal

logger = logging.getLogger(__name__)

DECISION_WATCH_INTERVAL_SECONDS = 180


class StrategyParams(BaseModel):
    window_seconds: int = Field(default=60, gt=0)
    price_delta_pct: Decimal = Field(default=Decimal("0.8"), gt=Decimal("0"))
    volume_spike_multiplier: Decimal = Field(default=Decimal("2.0"), gt=Decimal("0"))
    oi_min_increase_pct: Decimal = Field(default=Decimal("0.05"), ge=Decimal("0"))
    regime_coin: str = Field(default="BTC")
    ema_period: int = Field(default=20, gt=0)
    regime_buffer_pct: Decimal = Field(default=Decimal("0.15"), ge=Decimal("0"))
    min_trades_in_window: int = Field(default=3, gt=0)

    @classmethod
    def from_settings(cls, settings: HyperliquidSettings) -> StrategyParams:
        return cls(
            window_seconds=settings.strategy_window_seconds,
            price_delta_pct=settings.strategy_price_delta_pct,
            volume_spike_multiplier=settings.strategy_volume_spike_multiplier,
            oi_min_increase_pct=settings.strategy_min_oi_increase_pct,
            regime_coin=settings.strategy_regime_coin.strip().upper(),
            ema_period=settings.strategy_ema_period,
            regime_buffer_pct=settings.strategy_regime_buffer_pct,
            min_trades_in_window=settings.strategy_min_trades_in_window,
        )


class MarketRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class _TradeSample:
    ts: datetime
    px: Decimal
    notional: Decimal


@dataclass(slots=True)
class _BreakoutCandidate:
    coin: str
    side: SignalSide
    ref_px: Decimal
    current_px: Decimal
    delta_pct: Decimal
    window_volume: Decimal
    avg_volume: Decimal
    oi_at_breakout: Decimal
    detected_at: datetime


@dataclass
class _CoinState:
    trades: deque[_TradeSample] = field(default_factory=deque)
    mark_px: Decimal | None = None
    open_interest: Decimal | None = None
    prev_open_interest: Decimal | None = None
    volume_history: deque[Decimal] = field(default_factory=lambda: deque(maxlen=20))
    pending_breakout: _BreakoutCandidate | None = None


def compute_ema(closes: list[Decimal], period: int) -> Decimal:
    if len(closes) < period:
        raise ValueError(f"Need at least {period} closes to compute EMA{period}, got {len(closes)}")

    multiplier = Decimal(2) / Decimal(period + 1)
    ema = sum(closes[:period]) / Decimal(period)
    for close in closes[period:]:
        ema = (close * multiplier) + (ema * (Decimal(1) - multiplier))
    return ema


class _RegimeTracker:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._mark_prices: deque[Decimal] = deque(maxlen=max(params.ema_period * 3, params.ema_period))
        self._committed_regime: MarketRegime = MarketRegime.NEUTRAL

    def update(self, mark_px: Decimal) -> None:
        self._mark_prices.append(mark_px)

    def regime(self) -> MarketRegime:
        if len(self._mark_prices) < self.params.ema_period:
            return MarketRegime.NEUTRAL

        closes = list(self._mark_prices)
        ema = compute_ema(closes, self.params.ema_period)
        current = closes[-1]
        buffer = self.params.regime_buffer_pct / Decimal("100")
        bull_threshold = ema * (Decimal("1") + buffer)
        bear_threshold = ema * (Decimal("1") - buffer)

        if current > bull_threshold:
            self._committed_regime = MarketRegime.BULLISH
        elif current < bear_threshold:
            self._committed_regime = MarketRegime.BEARISH

        return self._committed_regime

    def regime_snapshot(self) -> dict[str, Decimal | str] | None:
        if len(self._mark_prices) < self.params.ema_period:
            return None
        closes = list(self._mark_prices)
        ema = compute_ema(closes, self.params.ema_period)
        current = closes[-1]
        distance_pct = ((current - ema) / ema) * Decimal("100") if ema > 0 else Decimal("0")
        return {
            "mark": current,
            "ema": ema,
            "distance_pct": distance_pct,
            "regime": self._committed_regime.value,
        }


class MomentumOIStrategy:
    """Dual-direction momentum + volume breakout strategy with OI confirmation."""

    def __init__(
        self,
        settings: HyperliquidSettings,
        params: StrategyParams | None = None,
        on_signal: Callable[[TradeSignal], None] | None = None,
        breadth: MarketBreadthTracker | None = None,
    ) -> None:
        self.settings = settings
        self.params = params or StrategyParams.from_settings(settings)
        self._on_signal = on_signal
        self._breadth = breadth
        self._coins: dict[str, _CoinState] = {}
        self._regime = _RegimeTracker(self.params)
        self._last_logged_regime: MarketRegime | None = None
        self._last_decision_watch: dict[str, datetime] = {}

    @staticmethod
    def _move_direction(side: SignalSide) -> str:
        return UP if side == SignalSide.LONG else DOWN

    def _breadth_record(self, coin: str, side: SignalSide, now: datetime) -> None:
        if self._breadth is not None:
            self._breadth.record(coin, self._move_direction(side), now)

    def _breadth_blocks(self, coin: str, side: SignalSide, now: datetime) -> bool:
        if self._breadth is None:
            return False
        return self._breadth.is_synchronized(
            self._move_direction(side), now, exclude=coin
        )

    def decision_summary(self, watchlist: tuple[str, ...], *, now: datetime | None = None) -> str:
        ts = now or self._utc_now()
        snap = self._regime.regime_snapshot()
        if snap is None:
            regime_part = f"regime=warming_up({self.params.regime_coin} EMA{self.params.ema_period})"
        else:
            regime_part = (
                f"regime={snap['regime']} | {self.params.regime_coin} "
                f"mark={snap['mark']} ema={snap['ema']:.2f} ({snap['distance_pct']:+.2f}% vs EMA)"
            )

        coin_parts: list[str] = []
        for symbol in watchlist:
            coin = symbol.strip().upper()
            state = self._coins.get(coin)
            if state is None:
                coin_parts.append(f"{coin}:idle")
                continue
            coin_parts.append(f"{coin}:{self._diagnose_coin(coin, state, ts)}")

        open_count = sum(1 for c in watchlist if self._coins.get(c.strip().upper()) and len(self._coins[c.strip().upper()].trades) > 0)
        pending = self.pending_breakout_coins()
        pending_part = ",".join(pending) if pending else "none"
        return f"{regime_part} | active_feeds={open_count}/{len(watchlist)} | pending_oi={pending_part} | {' · '.join(coin_parts)}"

    def pending_breakout_coins(self) -> tuple[str, ...]:
        return tuple(
            coin for coin, state in self._coins.items() if state.pending_breakout is not None
        )

    def _log_decision_watch(self, coin: str, now: datetime, message: str) -> None:
        last = self._last_decision_watch.get(coin)
        if last is not None and (now - last).total_seconds() < DECISION_WATCH_INTERVAL_SECONDS:
            return
        logger.info("Decision watch %s | %s", coin, message)
        self._last_decision_watch[coin] = now

    def _diagnose_coin(self, coin: str, state: _CoinState, now: datetime) -> str:
        if state.pending_breakout is not None:
            pending = state.pending_breakout
            oi_hint = ""
            if state.open_interest is not None and pending.oi_at_breakout > 0:
                oi_change = self._oi_change_pct(state.open_interest, pending.oi_at_breakout)
                oi_hint = f" oi={oi_change:.3f}% need>={self.params.oi_min_increase_pct}%"
            return f"pending {pending.side.value.upper()} delta={pending.delta_pct:.2f}%{oi_hint}"

        if state.mark_px is None:
            return "no mark yet"
        if state.prev_open_interest is None:
            return "waiting OI baseline"

        self._prune_trades(state, now)
        trade_count = len(state.trades)
        if trade_count < self.params.min_trades_in_window:
            return f"trades {trade_count}/{self.params.min_trades_in_window} in {self.params.window_seconds}s"

        ref_px = state.trades[0].px
        delta_pct = abs((state.mark_px - ref_px) / ref_px) * Decimal("100") if ref_px > 0 else Decimal("0")
        if delta_pct < self.params.price_delta_pct:
            return f"delta {delta_pct:.3f}% < need {self.params.price_delta_pct}%"

        side = SignalSide.LONG if state.mark_px > ref_px else SignalSide.SHORT
        allowed, regime = self._regime_side_allowed(side)
        if not allowed:
            return f"{side.value} blocked by regime={regime.value}"

        window_volume = sum(sample.notional for sample in state.trades)
        if state.volume_history:
            avg_volume = sum(state.volume_history) / Decimal(len(state.volume_history))
            required = avg_volume * self.params.volume_spike_multiplier
            if window_volume < required:
                return f"volume {window_volume:.0f} < need {required:.0f}"

        return f"near breakout {side.value} delta={delta_pct:.2f}% awaiting trigger"

    def _maybe_log_decision_watch(self, coin: str, state: _CoinState, now: datetime) -> None:
        if state.pending_breakout is not None or len(state.trades) > 0:
            self._log_decision_watch(coin, now, self._diagnose_coin(coin, state, now))

    def _track_regime(self, coin: str) -> None:
        if coin != self.params.regime_coin.strip().upper():
            return
        regime = self._regime.regime()
        if regime == self._last_logged_regime:
            return
        if self._last_logged_regime is None:
            snap = self._regime.regime_snapshot()
            dist = f" ({snap['distance_pct']:+.2f}% vs EMA)" if snap else ""
            logger.info(
                "Market regime initialized: %s (%s EMA%d)%s",
                regime.value,
                self.params.regime_coin,
                self.params.ema_period,
                dist,
            )
        else:
            snap = self._regime.regime_snapshot()
            dist = f" | {self.params.regime_coin} {snap['distance_pct']:+.2f}% vs EMA" if snap else ""
            logger.info(
                "Market regime changed: %s -> %s (%s EMA%d)%s",
                self._last_logged_regime.value,
                regime.value,
                self.params.regime_coin,
                self.params.ema_period,
                dist,
            )
        self._last_logged_regime = regime

    def _arm_pending_breakout(self, state: _CoinState, candidate: _BreakoutCandidate) -> None:
        if state.pending_breakout is None:
            logger.info(
                "Breakout detected %s %s | delta=%.2f%% | awaiting OI confirmation",
                candidate.coin,
                candidate.side.value.upper(),
                candidate.delta_pct,
            )
        # Count the detection toward market breadth even before OI confirms, so a
        # synchronized wave of breakouts is visible by the time one confirms.
        self._breadth_record(candidate.coin, candidate.side, candidate.detected_at)
        state.pending_breakout = candidate

    def _publish_signal(self, signal: TradeSignal, oi_change: Decimal) -> TradeSignal:
        logger.info(
            "Signal confirmed %s %s | mark=%s | oi_change=%.4f%% | confidence=%.2f",
            signal.symbol,
            signal.side.value.upper(),
            signal.entry_mark_price,
            oi_change,
            signal.confidence,
        )
        if self._on_signal is not None:
            self._on_signal(signal)
        return signal

    def _coin_state(self, coin: str) -> _CoinState:
        normalized = coin.strip().upper()
        if normalized not in self._coins:
            self._coins[normalized] = _CoinState()
        return self._coins[normalized]

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _prune_trades(self, state: _CoinState, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.params.window_seconds)
        while state.trades and state.trades[0].ts < cutoff:
            state.trades.popleft()

    def _detect_breakout(
        self,
        coin: str,
        state: _CoinState,
        now: datetime,
    ) -> _BreakoutCandidate | None:
        self._prune_trades(state, now)

        if len(state.trades) < self.params.min_trades_in_window:
            return None
        if state.mark_px is None or state.mark_px <= 0:
            return None
        if state.prev_open_interest is None or state.prev_open_interest <= 0:
            return None

        ref_px = state.trades[0].px
        current_px = state.mark_px
        if ref_px <= 0:
            return None

        delta_pct = abs((current_px - ref_px) / ref_px) * Decimal("100")
        if delta_pct < self.params.price_delta_pct:
            return None

        window_volume = sum(sample.notional for sample in state.trades)
        if state.volume_history:
            avg_volume = sum(state.volume_history) / Decimal(len(state.volume_history))
            if window_volume < avg_volume * self.params.volume_spike_multiplier:
                return None
        else:
            avg_volume = window_volume

        if current_px > ref_px:
            side = SignalSide.LONG
        elif current_px < ref_px:
            side = SignalSide.SHORT
        else:
            return None

        state.volume_history.append(window_volume)

        return _BreakoutCandidate(
            coin=coin,
            side=side,
            ref_px=ref_px,
            current_px=current_px,
            delta_pct=delta_pct,
            window_volume=window_volume,
            avg_volume=avg_volume,
            oi_at_breakout=state.prev_open_interest,
            detected_at=now,
        )

    @staticmethod
    def _oi_change_pct(current_oi: Decimal, baseline_oi: Decimal) -> Decimal:
        if baseline_oi <= 0:
            return Decimal("0")
        return ((current_oi - baseline_oi) / baseline_oi) * Decimal("100")

    def _confirm_oi(self, state: _CoinState, candidate: _BreakoutCandidate) -> Decimal | None:
        if state.open_interest is None or state.open_interest <= 0:
            logger.debug("OI confirmation skipped for %s: no current OI", candidate.coin)
            return None

        oi_change = self._oi_change_pct(state.open_interest, candidate.oi_at_breakout)
        if oi_change < self.params.oi_min_increase_pct:
            logger.debug(
                "Fakeout rejected for %s: OI change %.4f%% (baseline=%s current=%s)",
                candidate.coin,
                oi_change,
                candidate.oi_at_breakout,
                state.open_interest,
            )
            return None
        return oi_change

    def _regime_side_allowed(self, side: SignalSide) -> tuple[bool, MarketRegime]:
        regime = self._regime.regime()
        if regime == MarketRegime.NEUTRAL:
            return False, regime
        if regime == MarketRegime.BULLISH and side != SignalSide.LONG:
            return False, regime
        if regime == MarketRegime.BEARISH and side != SignalSide.SHORT:
            return False, regime
        return True, regime

    def _regime_allows(self, side: SignalSide) -> bool:
        allowed, regime = self._regime_side_allowed(side)
        if not allowed:
            if regime == MarketRegime.NEUTRAL:
                logger.debug("Signal blocked: market regime is NEUTRAL")
            elif regime == MarketRegime.BULLISH:
                logger.debug("Signal blocked: BULLISH regime allows LONG only")
            else:
                logger.debug("Signal blocked: BEARISH regime allows SHORT only")
        return allowed

    def _build_signal(
        self,
        candidate: _BreakoutCandidate,
        oi_change_pct: Decimal,
        timestamp: datetime,
    ) -> TradeSignal:
        vol_ratio = (
            candidate.window_volume / candidate.avg_volume
            if candidate.avg_volume > 0
            else Decimal("1")
        )
        price_score = candidate.delta_pct / (self.params.price_delta_pct * Decimal("2"))
        volume_score = vol_ratio / (self.params.volume_spike_multiplier * Decimal("2"))
        oi_score = oi_change_pct / (self.params.oi_min_increase_pct * Decimal("5"))
        raw_confidence = (price_score + volume_score + oi_score) / Decimal("3")
        confidence = min(Decimal("1"), max(Decimal("0"), raw_confidence))

        return TradeSignal(
            symbol=candidate.coin,
            side=candidate.side,
            entry_mark_price=candidate.current_px,
            confidence=confidence,
            timestamp=timestamp,
            reason="momentum_oi_breakout_confirmed",
        )

    def evaluate(
        self,
        coin: str,
        *,
        regime: MarketRegime,
        delta_pct: Decimal,
        window_volume: Decimal,
        avg_volume: Decimal,
        open_interest: Decimal,
        prev_open_interest: Decimal,
        mark_px: Decimal,
        side: SignalSide,
        timestamp: datetime | None = None,
    ) -> TradeSignal | None:
        if delta_pct < self.params.price_delta_pct:
            return None
        if avg_volume <= 0 or window_volume < avg_volume * self.params.volume_spike_multiplier:
            return None
        if prev_open_interest <= 0 or open_interest <= 0:
            return None

        oi_change = self._oi_change_pct(open_interest, prev_open_interest)
        if oi_change < self.params.oi_min_increase_pct:
            return None

        if regime == MarketRegime.NEUTRAL:
            return None
        if regime == MarketRegime.BULLISH and side != SignalSide.LONG:
            return None
        if regime == MarketRegime.BEARISH and side != SignalSide.SHORT:
            return None

        candidate = _BreakoutCandidate(
            coin=coin.strip().upper(),
            side=side,
            ref_px=mark_px,
            current_px=mark_px,
            delta_pct=delta_pct,
            window_volume=window_volume,
            avg_volume=avg_volume,
            oi_at_breakout=prev_open_interest,
            detected_at=timestamp or self._utc_now(),
        )
        return self._build_signal(candidate, oi_change, candidate.detected_at)

    async def on_market_event(self, event: MarketEvent) -> TradeSignal | None:
        if event.coin is None:
            return None

        coin = event.coin.strip().upper()
        now = event.ts if event.ts.tzinfo is not None else event.ts.replace(tzinfo=timezone.utc)
        state = self._coin_state(coin)

        if event.kind == EventKind.ASSET_CTX and isinstance(event.payload, AssetCtxPayload):
            if state.open_interest is not None:
                state.prev_open_interest = state.open_interest
            state.mark_px = event.payload.mark_px
            state.open_interest = event.payload.open_interest

            if coin == self.params.regime_coin.strip().upper():
                self._regime.update(event.payload.mark_px)
                self._track_regime(coin)

            if state.pending_breakout is not None:
                candidate = state.pending_breakout
                if not self._regime_allows(candidate.side):
                    self._log_decision_watch(
                        coin,
                        now,
                        f"pending {candidate.side.value.upper()} cancelled — regime={self._regime.regime().value}",
                    )
                    state.pending_breakout = None
                    return None

                oi_change = self._confirm_oi(state, candidate)
                state.pending_breakout = None
                if oi_change is None:
                    self._log_decision_watch(
                        coin,
                        now,
                        self._diagnose_coin(coin, state, now),
                    )
                    return None

                self._breadth_record(coin, candidate.side, now)
                if self._breadth_blocks(coin, candidate.side, now):
                    self._log_decision_watch(
                        coin,
                        now,
                        f"{candidate.side.value} blocked by market breadth (correlated move)",
                    )
                    return None

                return self._publish_signal(self._build_signal(candidate, oi_change, now), oi_change)

            self._maybe_log_decision_watch(coin, state, now)
            candidate = self._detect_breakout(coin, state, now)
            if candidate is None:
                return None
            if not self._regime_allows(candidate.side):
                self._log_decision_watch(
                    coin,
                    now,
                    f"breakout {candidate.side.value.upper()} delta={candidate.delta_pct:.2f}% blocked by regime={self._regime.regime().value}",
                )
                return None

            oi_change = self._confirm_oi(state, candidate)
            if oi_change is None:
                self._arm_pending_breakout(state, candidate)
                return None

            self._breadth_record(coin, candidate.side, now)
            if self._breadth_blocks(coin, candidate.side, now):
                self._log_decision_watch(
                    coin,
                    now,
                    f"{candidate.side.value} blocked by market breadth (correlated move)",
                )
                return None

            return self._publish_signal(self._build_signal(candidate, oi_change, now), oi_change)

        if event.kind == EventKind.TRADE and isinstance(event.payload, TradePayload):
            notional = event.payload.px * event.payload.sz
            state.trades.append(
                _TradeSample(ts=now, px=event.payload.px, notional=notional)
            )
            if state.mark_px is None:
                state.mark_px = event.payload.px

            candidate = self._detect_breakout(coin, state, now)
            if candidate is None:
                return None
            if not self._regime_allows(candidate.side):
                return None

            oi_change = self._confirm_oi(state, candidate)
            if oi_change is None:
                self._arm_pending_breakout(state, candidate)
                return None

            self._breadth_record(coin, candidate.side, now)
            if self._breadth_blocks(coin, candidate.side, now):
                self._log_decision_watch(
                    coin,
                    now,
                    f"{candidate.side.value} blocked by market breadth (correlated move)",
                )
                return None

            return self._publish_signal(self._build_signal(candidate, oi_change, now), oi_change)

        return None

    async def consume(self, queue: asyncio.Queue[MarketEvent]) -> None:
        while True:
            event = await queue.get()
            try:
                await self.on_market_event(event)
            except Exception:
                logger.exception("Strategy failed processing market event")

    async def on_candle(
        self,
        coin: str,
        open_px: Decimal,
        high_px: Decimal,
        low_px: Decimal,
        close_px: Decimal,
        volume: Decimal,
    ) -> TradeSignal | None:
        state = self._coin_state(coin)
        now = self._utc_now()
        notional = close_px * volume
        state.trades.append(_TradeSample(ts=now, px=close_px, notional=notional))
        state.mark_px = close_px

        candidate = self._detect_breakout(coin.strip().upper(), state, now)
        if candidate is None or not self._regime_allows(candidate.side):
            return None

        oi_change = self._confirm_oi(state, candidate)
        if oi_change is None:
            state.pending_breakout = candidate
            return None

        return self._build_signal(candidate, oi_change, now)

    async def on_open_interest(
        self,
        coin: str,
        open_interest: Decimal,
        mark_px: Decimal,
    ) -> TradeSignal | None:
        state = self._coin_state(coin)
        if state.open_interest is not None:
            state.prev_open_interest = state.open_interest
        state.open_interest = open_interest
        state.mark_px = mark_px

        if coin.strip().upper() == self.params.regime_coin.strip().upper():
            self._regime.update(mark_px)

        if state.pending_breakout is None:
            return None

        candidate = state.pending_breakout
        if not self._regime_allows(candidate.side):
            state.pending_breakout = None
            return None

        oi_change = self._confirm_oi(state, candidate)
        state.pending_breakout = None
        if oi_change is None:
            return None

        return self._build_signal(candidate, oi_change, self._utc_now())

    @property
    def current_regime(self) -> MarketRegime:
        return self._regime.regime()

    def seed_regime_prices(self, mark_prices: list[Decimal]) -> None:
        for price in mark_prices:
            self._regime.update(price)
