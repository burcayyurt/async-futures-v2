"""Tests for the order-construction path that only runs with real money.

Every order the bot has ever placed was a dry run, so the wire format has never
been validated against a live exchange. Two failures here are silent and
expensive: a wrong asset index sends the order to a *different coin*, and a
number formatted in scientific notation is rejected outright.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.core.config import HyperliquidSettings
from src.exchange.hyperliquid_rest import HyperliquidRestClient, OrderRequest


@pytest.fixture
def client() -> HyperliquidRestClient:
    settings = HyperliquidSettings.from_env().model_copy(update={"bot_dry_run": True})
    return HyperliquidRestClient(settings)


def _meta(*names: str) -> dict:
    return {"universe": [{"name": n} for n in names]}


# ----------------------------------------------------------------- asset index


@pytest.mark.asyncio
async def test_initialize_maps_coins_to_their_universe_index(
    client: HyperliquidRestClient,
) -> None:
    client.post_info = AsyncMock(return_value=_meta("BTC", "ETH", "SOL"))
    await client.initialize()

    assert client._coin_asset("BTC") == 0
    assert client._coin_asset("ETH") == 1
    assert client._coin_asset("SOL") == 2


@pytest.mark.asyncio
async def test_nameless_universe_entries_do_not_shift_later_indices(
    client: HyperliquidRestClient,
) -> None:
    """A malformed entry must not renumber the assets after it.

    Indices are positions in the exchange's universe array. If a skipped entry
    shifted the ones behind it, every later coin would trade under a neighbour's
    id — an order for SOL would be placed on some other market.
    """
    client.post_info = AsyncMock(
        return_value={"universe": [{"name": "BTC"}, {"noname": True}, {"name": "SOL"}]}
    )
    await client.initialize()

    assert client._coin_asset("BTC") == 0
    assert client._coin_asset("SOL") == 2  # not 1


@pytest.mark.asyncio
async def test_unknown_coin_raises_instead_of_guessing(
    client: HyperliquidRestClient,
) -> None:
    client.post_info = AsyncMock(return_value=_meta("BTC"))
    await client.initialize()

    with pytest.raises(ValueError, match="Unknown coin"):
        client._coin_asset("DOGE")


@pytest.mark.asyncio
async def test_coin_lookup_is_case_and_space_insensitive(
    client: HyperliquidRestClient,
) -> None:
    client.post_info = AsyncMock(return_value=_meta("BTC", "ETH"))
    await client.initialize()

    assert client._coin_asset(" eth ") == 1


# --------------------------------------------------------------- wire numbers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100", "100"),
        ("1000", "1000"),          # normalize() yields 1E+3 before formatting
        ("10000000", "10000000"),
        ("0.5", "0.5"),
        ("0.00001", "0.00001"),
        ("6.4500", "6.45"),        # trailing zeros dropped
        ("0", "0"),
        ("0.000000010", "0.00000001"),
    ],
)
def test_wire_floats_never_use_scientific_notation(raw: str, expected: str) -> None:
    """Hyperliquid rejects exponent form; round numbers are where it creeps in."""
    formatted = HyperliquidRestClient._normalize_wire_float(raw)
    assert formatted == expected
    assert "E" not in formatted.upper()


def test_wire_float_preserves_value() -> None:
    for raw in ("1000", "0.00001", "6.4500", "123456.789"):
        assert Decimal(HyperliquidRestClient._normalize_wire_float(raw)) == Decimal(raw)


# ------------------------------------------------------------ order assembly


@pytest.mark.asyncio
async def test_place_order_builds_the_expected_wire_order(
    client: HyperliquidRestClient,
) -> None:
    client.post_info = AsyncMock(return_value=_meta("BTC", "ETH", "SOL"))
    await client.initialize()

    result = await client.place_order(
        OrderRequest(
            coin="sol",
            is_buy=True,
            sz="12.500",
            limit_px="140.0",
            reduce_only=False,
            tif="Alo",
        )
    )

    order = result["action"]["orders"][0]
    assert order["a"] == 2          # SOL's index, not a name
    assert order["b"] is True
    assert order["p"] == "140"
    assert order["s"] == "12.5"
    assert order["r"] is False
    assert order["t"] == {"limit": {"tif": "Alo"}}
    assert result["action"]["grouping"] == "na"


@pytest.mark.asyncio
async def test_exit_orders_are_reduce_only_ioc(client: HyperliquidRestClient) -> None:
    """A reduce-only flag that fails to propagate could open a reversed position."""
    client.post_info = AsyncMock(return_value=_meta("BTC"))
    await client.initialize()

    result = await client.place_order(
        OrderRequest(
            coin="BTC", is_buy=False, sz="0.1", limit_px="50000", reduce_only=True, tif="Ioc"
        )
    )

    order = result["action"]["orders"][0]
    assert order["r"] is True
    assert order["t"] == {"limit": {"tif": "Ioc"}}


@pytest.mark.asyncio
async def test_dry_run_never_signs_or_posts(client: HyperliquidRestClient) -> None:
    """Dry run must stop before the network, even with a wallet configured."""
    client.post_info = AsyncMock(return_value=_meta("BTC"))
    await client.initialize()
    client.post_exchange = AsyncMock()
    client.sign_action = AsyncMock()

    result = await client.place_order(
        OrderRequest(coin="BTC", is_buy=True, sz="1", limit_px="100")
    )

    assert result["status"] == "dry_run"
    client.post_exchange.assert_not_awaited()
    client.sign_action.assert_not_awaited()


# ------------------------------------------------------- rejection detection


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        ({"status": "dry_run", "action": {}}, True),
        ({"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}}, True),
        (
            {"status": "ok", "response": {"data": {"statuses": [{"filled": {"oid": 1}}]}}},
            True,
        ),
        ({"status": "ok", "response": {"type": "default"}}, True),
        # Routine with post-only entries: an Alo order that would cross.
        (
            {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {"error": "Order could not immediately match against any resting orders."}
                        ]
                    }
                },
            },
            False,
        ),
        ({"status": "err", "response": "Insufficient margin"}, False),
        ({"status": "unexpected"}, False),
        ({}, False),
    ],
)
def test_order_rejection_reads_business_level_errors(payload: dict, accepted: bool) -> None:
    from src.exchange.hyperliquid_rest import order_rejection

    assert (order_rejection(payload) is None) is accepted


def test_partial_status_list_with_one_error_is_a_rejection() -> None:
    from src.exchange.hyperliquid_rest import order_rejection

    payload = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 1}}, {"error": "boom"}]}},
    }
    assert order_rejection(payload) == "boom"
