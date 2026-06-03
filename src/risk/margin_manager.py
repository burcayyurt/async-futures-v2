from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_rest import HyperliquidRestClient

logger = logging.getLogger(__name__)

MIN_NOTIONAL_USD = Decimal("10.0")
MAX_MARGIN_USAGE = Decimal("0.95")
LIQUIDATION_BUFFER_FACTOR = Decimal("0.1")


class MarginSafetyError(ValueError):
    """Position size failed sanity checks."""


@dataclass(slots=True)
class MarginSnapshot:
    equity: Decimal
    used_margin: Decimal
    available_margin: Decimal
    leverage: int


class MarginManager:
    """Isolated margin and leverage calculation for Hyperliquid perpetuals."""

    def __init__(self, settings: HyperliquidSettings, rest: HyperliquidRestClient) -> None:
        self.settings = settings
        self.rest = rest

    def calculate_position_size(
        self,
        free_collateral: float,
        mark_price: float,
        risk_pct: float,
        leverage: int,
    ) -> float:
        collateral = Decimal(str(free_collateral))
        mark_px = Decimal(str(mark_price))
        risk = Decimal(str(risk_pct))
        lev = Decimal(str(leverage))

        if collateral <= 0:
            raise MarginSafetyError("free_collateral must be positive")
        if mark_px <= 0:
            raise MarginSafetyError("mark_price must be positive")
        if leverage < 1:
            raise MarginSafetyError("leverage must be at least 1")
        if risk <= 0 or risk > Decimal("100"):
            raise MarginSafetyError("risk_pct must be in (0, 100]")

        if self.settings.margin_mode == "cross":
            raise MarginSafetyError(
                "Cross margin mode rejected: entire account collateral is at liquidation risk"
            )

        risk_fraction = risk / Decimal("100")
        margin_to_use = collateral * risk_fraction
        notional_usd = margin_to_use * lev
        size_coins = notional_usd / mark_px

        if margin_to_use >= collateral * MAX_MARGIN_USAGE:
            raise MarginSafetyError(
                f"Allocated margin {margin_to_use} exceeds {MAX_MARGIN_USAGE * 100}% "
                f"of free collateral {collateral}"
            )
        if notional_usd < MIN_NOTIONAL_USD:
            raise MarginSafetyError(
                f"Notional value {notional_usd} is below Hyperliquid minimum {MIN_NOTIONAL_USD} USD"
            )
        if size_coins <= 0:
            raise MarginSafetyError("Calculated position size must be positive")

        logger.debug(
            "Position size: collateral=%s margin=%s notional=%s size=%s",
            collateral,
            margin_to_use,
            notional_usd,
            size_coins,
        )
        return float(size_coins)

    def compute_position_size(self, notional_usd: Decimal, mark_px: Decimal) -> Decimal:
        if mark_px <= 0:
            raise MarginSafetyError("mark_px must be positive")
        if notional_usd < MIN_NOTIONAL_USD:
            raise MarginSafetyError(
                f"Notional value {notional_usd} is below minimum {MIN_NOTIONAL_USD} USD"
            )
        return notional_usd / mark_px

    def compute_liquidation_buffer(
        self,
        entry_px: Decimal,
        mark_px: Decimal,
        leverage: int,
        is_long: bool,
    ) -> Decimal:
        if leverage < 1:
            raise MarginSafetyError("leverage must be at least 1")
        if mark_px <= 0:
            raise MarginSafetyError("mark_px must be positive")

        maintenance_distance = mark_px / Decimal(str(leverage))
        buffer = maintenance_distance * LIQUIDATION_BUFFER_FACTOR
        if is_long:
            return mark_px - buffer
        return mark_px + buffer

    async def fetch_margin_snapshot(self) -> MarginSnapshot:
        state = await self.rest.get_clearinghouse_state()
        margin_summary = state.get("marginSummary") or {}
        equity = Decimal(str(margin_summary.get("accountValue", "0")))
        used_margin = Decimal(str(margin_summary.get("totalMarginUsed", "0")))
        available_margin = equity - used_margin
        return MarginSnapshot(
            equity=equity,
            used_margin=used_margin,
            available_margin=available_margin,
            leverage=self.settings.leverage,
        )

    async def ensure_isolated_leverage(self, coin: str, leverage: int) -> None:
        await self.rest._ensure_margin_settings(coin.strip().upper())
