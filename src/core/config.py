from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(symbol.strip() for symbol in value.split(",") if symbol.strip())
    if not symbols:
        raise ValueError("BOT_SYMBOLS must contain at least one coin symbol")
    return symbols


class HyperliquidSettings(BaseSettings):
    """Hyperliquid perpetual futures bot configuration loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    wallet_address: str = Field(default="", validation_alias="WALLET_ADDRESS")
    agent_private_key: SecretStr = Field(default=SecretStr(""), validation_alias="AGENT_PRIVATE_KEY")

    api_url: str = Field(default="https://api.hyperliquid.xyz", validation_alias="HL_API_URL")
    ws_url: str = Field(default="wss://api.hyperliquid.xyz/ws", validation_alias="HL_WS_URL")
    testnet: bool = Field(default=False, validation_alias="HL_TESTNET")

    leverage: int = Field(default=10, ge=1, validation_alias="LEVERAGE")
    margin_mode: Literal["isolated", "cross"] = Field(default="isolated", validation_alias="MARGIN_MODE")
    symbols: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(
            "BTC",
            "ETH",
            "SOL",
            "NEAR",
            "AVAX",
            "DOGE",
            "LINK",
            "INJ",
            "SUI",
            "SEI",
            "APT",
        ),
        validation_alias="BOT_SYMBOLS",
    )
    bot_dry_run: bool = Field(default=True, validation_alias="BOT_DRY_RUN")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    trade_journal_path: str = Field(default="data/trades.jsonl", validation_alias="TRADE_JOURNAL_PATH")
    open_positions_path: str = Field(default="data/open_positions.json", validation_alias="OPEN_POSITIONS_PATH")
    # Maps each trade's config_id back to the parameters that produced it.
    config_registry_path: str = Field(
        default="data/config_registry.jsonl", validation_alias="CONFIG_REGISTRY_PATH"
    )
    # Run windows, used to measure how much of the observation span was covered.
    session_log_path: str = Field(
        default="data/sessions.jsonl", validation_alias="SESSION_LOG_PATH"
    )
    session_state_path: str = Field(
        default="data/session_state.json", validation_alias="SESSION_STATE_PATH"
    )

    record_events: bool = Field(default=False, validation_alias="RECORD_EVENTS")
    recordings_dir: str = Field(default="data/recordings", validation_alias="RECORDINGS_DIR")

    telegram_bot_token: SecretStr = Field(default=SecretStr(""), validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", validation_alias="TELEGRAM_CHAT_ID")
    telegram_poll_enabled: bool = Field(default=True, validation_alias="TELEGRAM_POLL_ENABLED")
    telegram_poll_interval_seconds: int = Field(default=3, gt=0, validation_alias="TELEGRAM_POLL_INTERVAL_SECONDS")

    max_drawdown_pct: Decimal = Field(default=Decimal("2.0"), gt=Decimal("0"), validation_alias="MAX_DRAWDOWN_PCT")
    trade_risk_pct: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        le=Decimal("100"),
        validation_alias="TRADE_RISK_PCT",
    )
    trailing_callback_pct: Decimal = Field(
        default=Decimal("0.0015"),
        # 0 disables the trail entirely, which the horizon-exit configuration
        # needs; a positive floor would make that configuration unreachable.
        ge=Decimal("0"),
        validation_alias="TRAILING_CALLBACK_PCT",
    )
    break_even_trigger_pct: Decimal = Field(
        default=Decimal("0.1"),
        ge=Decimal("0"),
        validation_alias="BREAK_EVEN_TRIGGER_PCT",
    )

    # --- Feed-health guards. A 502/blackout on the Hyperliquid WS leaves the
    # strategy running on stale data and the position manager unable to trail
    # stops. After a (re)connect we suppress new entries for a warm-up window,
    # and we refuse to fire a hard stop on the first gapped mark that arrives
    # after a data blackout (it would exit at a discontinuous price). ---
    ws_reconnect_entry_cooldown_seconds: int = Field(
        default=15, ge=0, validation_alias="WS_RECONNECT_ENTRY_COOLDOWN_SECONDS"
    )
    mark_staleness_gap_seconds: Decimal = Field(
        default=Decimal("5.0"), ge=Decimal("0"), validation_alias="MARK_STALENESS_GAP_SECONDS"
    )
    # Minimum time a freshly opened position must be held before a hard stop can
    # fire (kill-switch always overrides). Guards against 1-tick whipsaw exits.
    min_hold_seconds: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), validation_alias="MIN_HOLD_SECONDS"
    )
    # Close a position this many seconds after entry regardless of price. 0 = off.
    # The DVSLA signal's forward edge peaks around 30-60s and decays after, so a
    # horizon exit captures it where a trail cannot: a trail seeded at the entry
    # price truncates exactly the upside the signal predicts.
    max_hold_seconds: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), validation_alias="MAX_HOLD_SECONDS"
    )

    strategy_window_seconds: int = Field(default=60, gt=0, validation_alias="STRATEGY_WINDOW_SECONDS")
    strategy_price_delta_pct: Decimal = Field(
        default=Decimal("0.8"),
        gt=Decimal("0"),
        validation_alias="STRATEGY_PRICE_DELTA",
    )
    strategy_volume_spike_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        gt=Decimal("0"),
        validation_alias="STRATEGY_VOLUME_SPIKE",
    )
    strategy_min_oi_increase_pct: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0"),
        validation_alias="STRATEGY_MIN_OI_INCREASE",
    )
    strategy_regime_coin: str = Field(default="BTC", validation_alias="STRATEGY_REGIME_COIN")
    strategy_ema_period: int = Field(default=20, gt=0, validation_alias="STRATEGY_EMA_PERIOD")
    strategy_regime_buffer_pct: Decimal = Field(
        default=Decimal("0.15"),
        ge=Decimal("0"),
        validation_alias="STRATEGY_REGIME_BUFFER_PCT",
    )
    strategy_min_trades_in_window: int = Field(
        default=3,
        gt=0,
        validation_alias="STRATEGY_MIN_TRADES",
    )

    # --- Correlation-trap guard. When many distinct symbols move the same
    # direction inside a short window the move is market beta, not a per-symbol
    # edge; the engine skips the signal to avoid chasing a synchronized move. ---
    correlation_guard_enabled: bool = Field(
        default=True, validation_alias="CORRELATION_GUARD_ENABLED"
    )
    correlation_window_seconds: int = Field(
        default=120, gt=0, validation_alias="CORRELATION_WINDOW_SECONDS"
    )
    correlation_max_same_side: int = Field(
        default=4, gt=0, validation_alias="CORRELATION_MAX_SAME_SIDE"
    )

    # --- Execution upgrades (PR5). Opt-in; defaults preserve legacy behavior. ---
    maker_entry_enabled: bool = Field(default=False, validation_alias="MAKER_ENTRY_ENABLED")
    # Exchange fees in basis points, mirroring ``backtest.simulator.SimConfig`` so
    # live journal PnL and backtest PnL are directly comparable. Entries are maker
    # only when ``maker_entry_enabled`` (Alo/post-only); exits are always IOC
    # reduce-only, i.e. taker. Override per your Hyperliquid VIP tier.
    maker_fee_bps: Decimal = Field(
        default=Decimal("1.5"), ge=Decimal("0"), validation_alias="MAKER_FEE_BPS"
    )
    taker_fee_bps: Decimal = Field(
        default=Decimal("4.5"), ge=Decimal("0"), validation_alias="TAKER_FEE_BPS"
    )
    confidence_sizing_enabled: bool = Field(
        default=False, validation_alias="CONFIDENCE_SIZING_ENABLED"
    )
    confidence_size_floor: Decimal = Field(
        default=Decimal("0.25"),
        ge=Decimal("0"),
        le=Decimal("1"),
        validation_alias="CONFIDENCE_SIZE_FLOOR",
    )
    reentry_cooldown_seconds: int = Field(
        default=0, ge=0, validation_alias="REENTRY_COOLDOWN_SECONDS"
    )
    atr_stop_enabled: bool = Field(default=False, validation_alias="ATR_STOP_ENABLED")
    atr_window: int = Field(default=50, gt=1, validation_alias="ATR_WINDOW")
    atr_stop_mult: Decimal = Field(
        default=Decimal("2.5"), gt=Decimal("0"), validation_alias="ATR_STOP_MULT"
    )
    # Floor on the ATR-stop distance. The ATR proxy is built from per-tick mark
    # returns (asset_ctx prints arrive multiple times/sec), so its raw value is
    # microscopic (~0.01%); without a floor the stop sits on top of the entry and
    # any tick of noise triggers it. Keep stops at least this wide (fraction).
    # DVSLA's edge is small (~0.07-0.7% ROE), so the floor stays tight: ~0.4% is
    # still ~20x the per-tick noise but keeps risk:reward sane against the TP.
    atr_stop_min_pct: Decimal = Field(
        default=Decimal("0.004"), ge=Decimal("0"), validation_alias="ATR_STOP_MIN_PCT"
    )
    reversion_tp_enabled: bool = Field(
        default=False, validation_alias="REVERSION_TP_ENABLED"
    )
    reversion_window: int = Field(default=50, gt=1, validation_alias="REVERSION_WINDOW")

    # --- Strategy engine selection (PR6). DVSLA is the live engine; the legacy
    # momentum engine is retained only for A/B comparison and tests. ---
    strategy_engine: Literal["dvsla", "momentum"] = Field(
        default="dvsla", validation_alias="STRATEGY_ENGINE"
    )

    # --- DVSLA (liquidation-cascade mean-reversion) parameters. ---
    dvsla_volume_bar_threshold: Decimal = Field(
        default=Decimal("50"), gt=Decimal("0"), validation_alias="DVSLA_VOLUME_BAR_THRESHOLD"
    )
    # Per-symbol volume-bar size thresholds. Calibrated from recorded throughput
    # (events-2026-06-16) so every coin closes ~2 bars/min (raw traded size
    # differs by >1000x across the watchlist, so a single global threshold is
    # meaningless and the tiny fallback caused thousands of micro-bars/sec).
    # Override via the DVSLA_SYMBOL_THRESHOLDS env var as a JSON object,
    # e.g. {"BTC":"40"}. Re-sweep periodically as throughput drifts.
    dvsla_symbol_thresholds: dict[str, Decimal] = Field(
        default_factory=lambda: {
            "WLD": Decimal("69000"),
            "SEI": Decimal("36000"),
            "XRP": Decimal("25000"),
            "ADA": Decimal("13000"),
            "DOGE": Decimal("9400"),
            "NEAR": Decimal("4500"),
            "SUI": Decimal("2900"),
            "ARB": Decimal("2800"),
            "FET": Decimal("1800"),
            "TIA": Decimal("1600"),
            "OP": Decimal("1400"),
            "APT": Decimal("810"),
            "ETH": Decimal("400"),   # 640→400: warmup ~4sa→~2.5sa
            "ATOM": Decimal("420"),
            "SOL": Decimal("370"),
            "FIL": Decimal("300"),
            "INJ": Decimal("220"),
            "LINK": Decimal("150"),
            "AVAX": Decimal("130"),
            "DOT": Decimal("93"),
            "LTC": Decimal("11"),
            "BTC": Decimal("8"),
            "BCH": Decimal("3"),
            "BNB": Decimal("2"),
        },
        validation_alias="DVSLA_SYMBOL_THRESHOLDS",
    )
    dvsla_ret_z_window: int = Field(default=50, gt=1, validation_alias="DVSLA_RET_Z_WINDOW")
    dvsla_ret_z_entry: Decimal = Field(
        default=Decimal("3.0"), gt=Decimal("0"), validation_alias="DVSLA_RET_Z_ENTRY"
    )
    # Minimum absolute bar return (fraction) for a bar to qualify as a return
    # shock. Guards against variance-collapse blow-ups: a near-flat window can
    # drive the z-score to absurd values, so an extreme z alone is not enough —
    # the move itself must be at least this big (default 0.15%).
    dvsla_ret_min_abs_pct: Decimal = Field(
        default=Decimal("0.0015"), ge=Decimal("0"), validation_alias="DVSLA_RET_MIN_ABS_PCT"
    )
    # Hard cap on |ret_z| fed to the signal (collapsed-variance windows can
    # otherwise yield z-scores in the millions).
    dvsla_ret_z_clamp: Decimal = Field(
        default=Decimal("25"), gt=Decimal("0"), validation_alias="DVSLA_RET_Z_CLAMP"
    )
    # Reject entries whose |ret_z| is at/above this level. A z-score near the
    # clamp ceiling is not a real cascade but a variance-collapse or post-
    # reconnect burst artefact (the single largest loser in the invert-era
    # dry-run had ret_z=25.00). 0 disables the filter. Keep below ret_z_clamp.
    dvsla_ret_z_reject: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), validation_alias="DVSLA_RET_Z_REJECT"
    )
    dvsla_flow_window: int = Field(default=200, gt=0, validation_alias="DVSLA_FLOW_WINDOW")
    dvsla_flow_imbalance_min: Decimal = Field(
        default=Decimal("0.5"),
        ge=Decimal("0"),
        le=Decimal("1"),
        validation_alias="DVSLA_FLOW_IMBALANCE_MIN",
    )
    dvsla_oi_z_window: int = Field(default=50, gt=1, validation_alias="DVSLA_OI_Z_WINDOW")
    dvsla_oi_z_drop: Decimal = Field(
        default=Decimal("-1.0"), lt=Decimal("0"), validation_alias="DVSLA_OI_Z_DROP"
    )
    # When fading (invert=False), a strongly *positive* OI z-score means open
    # interest is expanding on the move — fresh directional money, not a
    # liquidation cascade. Fading that is catching a knife against a real trend,
    # so veto the signal when oi_z >= |oi_z_drop| (only applies in fade mode).
    dvsla_fade_oi_z_veto: bool = Field(
        default=True, validation_alias="DVSLA_FADE_OI_Z_VETO"
    )
    dvsla_hurst_window: int = Field(default=64, gt=16, validation_alias="DVSLA_HURST_WINDOW")
    dvsla_hurst_max: Decimal = Field(
        default=Decimal("0.55"),
        gt=Decimal("0"),
        lt=Decimal("1"),
        validation_alias="DVSLA_HURST_MAX",
    )
    dvsla_warmup_bars: int = Field(default=20, ge=0, validation_alias="DVSLA_WARMUP_BARS")
    dvsla_cooldown_bars: int = Field(default=10, ge=0, validation_alias="DVSLA_COOLDOWN_BARS")
    # Reject signals whose confidence falls below this floor. Confidence measures
    # agreement across the engine's confirmations, and on the full recording set
    # the relationship to per-trade edge is monotonic: raising the floor lifts
    # win rate and cuts the ordinary-day bleed that the rare cascades have to pay
    # for. The gate belongs here rather than inside the strategy so a rejected
    # signal still consumes its cooldown — the same order the sweep measured.
    # 0 disables. See scripts/sweep_entry_threshold.py.
    dvsla_min_confidence: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        validation_alias="DVSLA_MIN_CONFIDENCE",
    )
    # Trade direction. The original thesis fades the cascade (mean-reversion);
    # recorded-data backtests show the cascades *continue* on the FAZ-0-Lite feed,
    # so inverting the side (trade WITH the cascade = momentum) is net-positive.
    # Default False preserves the legacy fade behaviour; flip via DVSLA_INVERT.
    dvsla_invert: bool = Field(default=False, validation_alias="DVSLA_INVERT")

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            return _parse_symbols(value)
        raise TypeError("BOT_SYMBOLS must be a comma-separated string")

    @field_validator("testnet", "bot_dry_run", "telegram_poll_enabled", "record_events", "maker_entry_enabled", "confidence_sizing_enabled", "atr_stop_enabled", "reversion_tp_enabled", "dvsla_invert", "dvsla_fade_oi_z_veto", "correlation_guard_enabled", mode="before")
    @classmethod
    def parse_bool(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def from_env(cls) -> HyperliquidSettings:
        return cls()

    @property
    def info_url(self) -> str:
        return f"{self.api_url.rstrip('/')}/info"

    @property
    def exchange_url(self) -> str:
        return f"{self.api_url.rstrip('/')}/exchange"

    def require_credentials(self) -> None:
        if not self.wallet_address.strip():
            raise ValueError("WALLET_ADDRESS is required for live trading")
        if not self.agent_private_key.get_secret_value().strip():
            raise ValueError("AGENT_PRIVATE_KEY is required for live trading")
