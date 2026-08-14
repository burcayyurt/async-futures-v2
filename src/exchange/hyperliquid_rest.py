from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import aiohttp
import msgpack
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_hex

from src.core.config import HyperliquidSettings

logger = logging.getLogger(__name__)

TimeInForce = Literal["Alo", "Ioc", "Gtc"]

VERIFYING_CONTRACT = "0x0000000000000000000000000000000000000000"
PHANTOM_DOMAIN = {
    "name": "Exchange",
    "version": "1",
    "chainId": 1337,
    "verifyingContract": VERIFYING_CONTRACT,
}
AGENT_TYPES = {
    "Agent": [
        {"name": "source", "type": "string"},
        {"name": "connectionId", "type": "bytes32"},
    ],
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
}


@dataclass(slots=True)
class OrderRequest:
    coin: str
    is_buy: bool
    sz: str
    limit_px: str
    reduce_only: bool = False
    tif: TimeInForce = "Gtc"


@dataclass(slots=True)
class SignedPayload:
    """EIP-712 signed action ready for POST /exchange."""

    action: dict[str, Any]
    nonce: int
    signature: dict[str, Any]


def order_rejection(result: dict[str, Any]) -> str | None:
    """Return the rejection reason for an order response, or ``None`` if accepted.

    Hyperliquid signals business-level failures inside an HTTP 200 body, so
    ``raise_for_status`` never sees them. Two shapes matter::

        {"status": "err", "response": "Insufficient margin"}
        {"status": "ok", "response": {"data": {"statuses": [{"error": "..."}]}}}

    The second is routine with post-only entries: an ``Alo`` order that would
    cross is refused rather than filled. Treating that as success registers a
    position that does not exist.

    Anything unrecognised counts as a rejection. For an entry that means not
    opening; for an exit it means keeping the position under management. Both
    are the safe direction when the alternative is losing track of real money.
    """

    status = result.get("status")
    if status == "dry_run":
        return None
    if status == "err":
        return str(result.get("response", "exchange returned an error"))
    if status != "ok":
        return f"unrecognised order response status: {status!r}"

    response = result.get("response")
    if not isinstance(response, dict):
        return f"unrecognised order response payload: {response!r}"
    data = response.get("data")
    if not isinstance(data, dict):
        # Some acknowledgements carry no data block; treat as accepted.
        return None
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or not statuses:
        return None
    for entry in statuses:
        if isinstance(entry, dict) and "error" in entry:
            return str(entry["error"])
    return None


@dataclass(frozen=True, slots=True)
class ExchangePosition:
    """A position as the exchange reports it — the source of truth on restart."""

    coin: str
    signed_size: Decimal
    entry_px: Decimal

    @property
    def is_long(self) -> bool:
        return self.signed_size > 0

    @property
    def abs_size(self) -> Decimal:
        return abs(self.signed_size)


class HyperliquidRestClient:
    """
    REST client for Hyperliquid /info and /exchange endpoints.

    L1 actions use phantom-agent EIP-712 signing. Domain, types, and wallet
    are cached once in __init__; the hot path only hashes the dynamic action.
    """

    def __init__(self, settings: HyperliquidSettings) -> None:
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._coin_to_asset: dict[str, int] = {}
        self._initialized = False
        self._leverage_applied: set[str] = set()

        private_key = settings.agent_private_key.get_secret_value().strip()
        if private_key:
            self._wallet = Account.from_key(private_key)
        else:
            self._wallet = None

        self._is_mainnet = not settings.testnet
        self._phantom_source = "a" if self._is_mainnet else "b"
        self._cached_domain = PHANTOM_DOMAIN
        self._cached_types = AGENT_TYPES
        self._vault_address: str | None = None
        self._expires_after: int | None = None

    async def initialize(self) -> None:
        """Fetch meta and build coin-to-asset index map."""
        if self._initialized:
            return

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )

        meta = await self.post_info({"type": "meta"})
        universe = meta.get("universe") or []
        self._coin_to_asset = {
            str(asset["name"]): index for index, asset in enumerate(universe) if "name" in asset
        }
        self._initialized = True
        logger.info("Hyperliquid meta loaded for %d assets", len(self._coin_to_asset))

    def sign_action(self, action: dict[str, Any], nonce: int) -> SignedPayload:
        signature = self._sign_l1_action_sync(action, nonce)
        return SignedPayload(action=action, nonce=nonce, signature=signature)

    async def post_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_session()
        assert self._session is not None
        async with self._session.post(self.settings.info_url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict from /info, got {type(data)!r}")
            return data

    async def post_exchange(self, signed: SignedPayload) -> dict[str, Any]:
        await self._ensure_session()
        assert self._session is not None
        body = {
            "action": signed.action,
            "nonce": signed.nonce,
            "signature": signed.signature,
            "vaultAddress": self._vault_address,
            "expiresAfter": self._expires_after,
        }
        async with self._session.post(self.settings.exchange_url, json=body) as response:
            response.raise_for_status()
            data = await response.json()
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict from /exchange, got {type(data)!r}")
            return data

    async def get_clearinghouse_state(self) -> dict[str, Any]:
        wallet = self.settings.wallet_address.strip()
        if not wallet:
            raise ValueError("WALLET_ADDRESS is required to read positions")
        return await self.post_info({"type": "clearinghouseState", "user": wallet})

    async def get_open_positions(self) -> dict[str, ExchangePosition]:
        """Return live positions keyed by coin, skipping fully-closed legs.

        Hyperliquid reports size as ``szi``: signed, negative for shorts. A zero
        ``szi`` means the leg is flat and is omitted rather than reported as a
        position with no size.
        """

        state = await self.get_clearinghouse_state()
        positions: dict[str, ExchangePosition] = {}
        for entry in state.get("assetPositions") or []:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("position")
            if not isinstance(raw, dict):
                continue
            coin = str(raw.get("coin", "")).strip().upper()
            if not coin:
                continue
            try:
                signed_size = Decimal(str(raw.get("szi", "0")))
                entry_px = Decimal(str(raw.get("entryPx") or "0"))
            except (ArithmeticError, ValueError):
                logger.warning("Unparseable position payload for %s: %r", coin, raw)
                continue
            if signed_size == 0:
                continue
            positions[coin] = ExchangePosition(
                coin=coin, signed_size=signed_size, entry_px=entry_px
            )
        return positions

    async def place_order(self, order: OrderRequest) -> dict[str, Any]:
        await self._require_initialized()
        coin = order.coin.strip().upper()
        await self._ensure_margin_settings(coin)

        order_wire = {
            "a": self._coin_asset(coin),
            "b": order.is_buy,
            "p": self._normalize_wire_float(order.limit_px),
            "s": self._normalize_wire_float(order.sz),
            "r": order.reduce_only,
            "t": {"limit": {"tif": order.tif}},
        }
        action = {"type": "order", "orders": [order_wire], "grouping": "na"}

        if self.settings.bot_dry_run:
            logger.info("[DRY RUN] place_order action=%s", action)
            return {"status": "dry_run", "action": action}

        self._require_wallet()
        nonce = self._timestamp_ms()
        signed = await asyncio.to_thread(self.sign_action, action, nonce)
        return await self.post_exchange(signed)

    async def cancel_order(self, coin: str, oid: int) -> dict[str, Any]:
        await self._require_initialized()
        normalized = coin.strip().upper()
        action = {
            "type": "cancel",
            "cancels": [{"a": self._coin_asset(normalized), "o": oid}],
        }

        if self.settings.bot_dry_run:
            logger.info("[DRY RUN] cancel_order action=%s", action)
            return {"status": "dry_run", "action": action}

        self._require_wallet()
        nonce = self._timestamp_ms()
        signed = await asyncio.to_thread(self.sign_action, action, nonce)
        return await self.post_exchange(signed)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )

    async def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call await initialize() before using HyperliquidRestClient")

    def _require_wallet(self) -> None:
        if self._wallet is None:
            raise ValueError("AGENT_PRIVATE_KEY is required for signed exchange actions")

    def _coin_asset(self, coin: str) -> int:
        normalized = coin.strip().upper()
        try:
            return self._coin_to_asset[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown coin symbol: {normalized}") from exc

    async def _ensure_margin_settings(self, coin: str) -> None:
        normalized = coin.strip().upper()
        if normalized in self._leverage_applied:
            return

        is_cross = self.settings.margin_mode == "cross"
        action = {
            "type": "updateLeverage",
            "asset": self._coin_asset(normalized),
            "isCross": is_cross,
            "leverage": self.settings.leverage,
        }

        if self.settings.bot_dry_run:
            logger.info("[DRY RUN] updateLeverage action=%s", action)
            self._leverage_applied.add(normalized)
            return

        self._require_wallet()
        nonce = self._timestamp_ms()
        signed = await asyncio.to_thread(self.sign_action, action, nonce)
        result = await self.post_exchange(signed)
        logger.info("Applied leverage settings for %s: %s", normalized, result)
        self._leverage_applied.add(normalized)

    def _action_hash(self, action: dict[str, Any], nonce: int) -> bytes:
        data = msgpack.packb(action)
        data += nonce.to_bytes(8, "big")
        if self._vault_address is None:
            data += b"\x00"
        else:
            data += b"\x01"
            data += self._address_to_bytes(self._vault_address)
        if self._expires_after is not None:
            data += b"\x00"
            data += self._expires_after.to_bytes(8, "big")
        return keccak(data)

    def _sign_l1_action_sync(self, action: dict[str, Any], nonce: int) -> dict[str, Any]:
        self._require_wallet()
        connection_id = self._action_hash(action, nonce)
        phantom_agent = {
            "source": self._phantom_source,
            "connectionId": connection_id,
        }
        full_message = {
            "domain": self._cached_domain,
            "types": self._cached_types,
            "primaryType": "Agent",
            "message": phantom_agent,
        }
        structured_data = encode_typed_data(full_message=full_message)
        assert self._wallet is not None
        signed = self._wallet.sign_message(structured_data)
        return {"r": to_hex(signed.r), "s": to_hex(signed.s), "v": signed.v}

    @staticmethod
    def _address_to_bytes(address: str) -> bytes:
        normalized = address[2:] if address.startswith("0x") else address
        return bytes.fromhex(normalized)

    @staticmethod
    def _normalize_wire_float(value: str) -> str:
        normalized = Decimal(value).normalize()
        return format(normalized, "f")

    @staticmethod
    def _timestamp_ms() -> int:
        return int(time.time() * 1000)
