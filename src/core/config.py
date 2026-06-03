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
        default=Decimal("0.006"),
        gt=Decimal("0"),
        validation_alias="TRAILING_CALLBACK_PCT",
    )
    break_even_trigger_pct: Decimal = Field(
        default=Decimal("0.5"),
        ge=Decimal("0"),
        validation_alias="BREAK_EVEN_TRIGGER_PCT",
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

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            return _parse_symbols(value)
        raise TypeError("BOT_SYMBOLS must be a comma-separated string")

    @field_validator("testnet", "bot_dry_run", "telegram_poll_enabled", mode="before")
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
