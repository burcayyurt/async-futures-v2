from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import HyperliquidSettings
from src.risk.kill_switch import KillSwitch
from src.risk.margin_manager import MarginManager, MarginSafetyError


@pytest.fixture
def isolated_settings() -> HyperliquidSettings:
    return HyperliquidSettings.from_env().model_copy(update={"margin_mode": "isolated"})


@pytest.fixture
def cross_settings() -> HyperliquidSettings:
    return HyperliquidSettings.from_env().model_copy(update={"margin_mode": "cross"})


@pytest.fixture
def margin_manager(isolated_settings: HyperliquidSettings) -> MarginManager:
    rest = MagicMock()
    return MarginManager(isolated_settings, rest)


def test_calculate_position_size_basic(margin_manager: MarginManager) -> None:
    size = margin_manager.calculate_position_size(
        free_collateral=1000.0,
        mark_price=50000.0,
        risk_pct=2.0,
        leverage=10,
    )
    assert size == pytest.approx(0.004)


def test_calculate_position_size_rejects_cross_margin(
    cross_settings: HyperliquidSettings,
) -> None:
    manager = MarginManager(cross_settings, MagicMock())
    with pytest.raises(MarginSafetyError, match="Cross margin"):
        manager.calculate_position_size(
            free_collateral=1000.0,
            mark_price=50000.0,
            risk_pct=2.0,
            leverage=10,
        )


def test_calculate_position_size_rejects_below_min_notional(
    margin_manager: MarginManager,
) -> None:
    with pytest.raises(MarginSafetyError, match="below Hyperliquid minimum"):
        margin_manager.calculate_position_size(
            free_collateral=100.0,
            mark_price=50000.0,
            risk_pct=0.01,
            leverage=1,
        )


async def test_kill_switch_debounce_dismisses_spike(isolated_settings: HyperliquidSettings) -> None:
    kill_switch = KillSwitch(isolated_settings)
    mark_price = Decimal("50000")

    await kill_switch.check_drawdown(Decimal("1000"), mark_price, coin="BTC")
    assert kill_switch.is_tripped is False

    async def recheck(_mark: Decimal) -> Decimal:
        return Decimal("990")

    with patch("src.risk.kill_switch.DRAWDOWN_DEBOUNCE_SECONDS", 0.01):
        tripped = await kill_switch.check_drawdown(
            Decimal("970"),
            mark_price,
            coin="BTC",
            recheck_equity=recheck,
        )

    assert tripped is False
    assert kill_switch.is_tripped is False


async def test_kill_switch_trips_after_confirmed_drawdown(
    isolated_settings: HyperliquidSettings,
) -> None:
    kill_switch = KillSwitch(isolated_settings)
    mark_price = Decimal("50000")

    await kill_switch.check_drawdown(Decimal("1000"), mark_price, coin="BTC")

    async def recheck(_mark: Decimal) -> Decimal:
        return Decimal("970")

    with patch("src.risk.kill_switch.DRAWDOWN_DEBOUNCE_SECONDS", 0.01):
        tripped = await kill_switch.check_drawdown(
            Decimal("970"),
            mark_price,
            coin="BTC",
            recheck_equity=recheck,
        )

    assert tripped is True
    assert kill_switch.is_tripped is True
    assert kill_switch.trip_reason is not None


async def test_fetch_margin_snapshot(margin_manager: MarginManager) -> None:
    margin_manager.rest.get_clearinghouse_state = AsyncMock(
        return_value={
            "marginSummary": {
                "accountValue": "1000.0",
                "totalMarginUsed": "200.0",
            }
        }
    )

    snapshot = await margin_manager.fetch_margin_snapshot()

    assert snapshot.equity == Decimal("1000.0")
    assert snapshot.used_margin == Decimal("200.0")
    assert snapshot.available_margin == Decimal("800.0")
