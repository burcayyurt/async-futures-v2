from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from src.core.config import HyperliquidSettings

logger = logging.getLogger(__name__)

MarketTickCallback = Callable[["MarketTick"], Awaitable[None]]

DEFAULT_QUEUE_MAXSIZE = 10_000
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0


class EventKind(str, Enum):
    TRADE = "trade"
    ASSET_CTX = "asset_ctx"
    WEB_DATA = "web_data"


@dataclass(slots=True)
class TradePayload:
    px: Decimal
    sz: Decimal
    side: str
    hash: str
    tid: int | None


@dataclass(slots=True)
class AssetCtxPayload:
    mark_px: Decimal
    open_interest: Decimal
    funding: Decimal | None
    oracle_px: Decimal | None
    day_ntl_vlm: Decimal | None


@dataclass(slots=True)
class MarketEvent:
    kind: EventKind
    coin: str | None
    ts: datetime
    payload: TradePayload | AssetCtxPayload | dict[str, Any]


@dataclass(slots=True)
class MarketTick:
    coin: str
    mark_px: Decimal
    funding: Decimal | None
    ts: datetime


@dataclass(frozen=True, slots=True)
class _Subscription:
    channel_type: str
    coin: str | None = None
    user: str | None = None

    def to_message(self) -> dict[str, Any]:
        subscription: dict[str, Any] = {"type": self.channel_type}
        if self.coin is not None:
            subscription["coin"] = self.coin
        if self.user is not None:
            subscription["user"] = self.user
        return {"method": "subscribe", "subscription": subscription}


class HyperliquidWebSocketListener:
    """Async WebSocket listener for Hyperliquid market and user feeds."""

    def __init__(
        self,
        settings: HyperliquidSettings,
        queue: asyncio.Queue[MarketEvent] | None = None,
        on_tick: MarketTickCallback | None = None,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self.settings = settings
        self._queue = queue or asyncio.Queue(maxsize=queue_maxsize)
        self._on_tick = on_tick
        self._connected = False
        self._ws: ClientConnection | None = None
        self._stop_event = asyncio.Event()
        self._subscriptions: set[_Subscription] = set()
        self._backoff_seconds = BACKOFF_BASE_SECONDS

        for coin in settings.symbols:
            self._register_symbol_subscriptions(coin)

        if settings.wallet_address.strip():
            self._subscriptions.add(
                _Subscription(channel_type="webData2", user=settings.wallet_address.strip())
            )

    @property
    def queue(self) -> asyncio.Queue[MarketEvent]:
        return self._queue

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _register_symbol_subscriptions(self, coin: str) -> None:
        normalized = coin.strip().upper()
        if not normalized:
            return
        self._subscriptions.add(_Subscription(channel_type="trades", coin=normalized))
        self._subscriptions.add(_Subscription(channel_type="activeAssetCtx", coin=normalized))

    async def connect(self) -> None:
        if self._connected and self._ws is not None:
            return
        self._ws = await websockets.connect(
            self.settings.ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        )
        self._connected = True
        self._backoff_seconds = BACKOFF_BASE_SECONDS
        await self._send_all_subscriptions()
        logger.info(
            "WebSocket connected | url=%s symbols=%d subscriptions=%d",
            self.settings.ws_url,
            len(self.settings.symbols),
            len(self._subscriptions),
        )

    async def disconnect(self) -> None:
        self._stop_event.set()
        self._connected = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def subscribe_trades(self, coin: str) -> None:
        normalized = coin.strip().upper()
        subscription = _Subscription(channel_type="trades", coin=normalized)
        self._subscriptions.add(subscription)
        if self._connected and self._ws is not None:
            await self._ws.send(json.dumps(subscription.to_message()))

    async def subscribe_l2(self, coin: str) -> None:
        raise NotImplementedError("l2Book subscription is not implemented in FAZ 2")

    async def subscribe_user_fills(self) -> None:
        raise NotImplementedError("userFills subscription is not implemented in FAZ 2")

    async def run(self) -> None:
        """Main listen loop with exponential-backoff auto-reconnect."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                await self.connect()
                assert self._ws is not None
                async for raw_message in self._ws:
                    if self._stop_event.is_set():
                        break
                    await self._handle_message(raw_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("WebSocket connection error; reconnecting")
            finally:
                self._connected = False
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        logger.debug("Error while closing websocket", exc_info=True)
                    self._ws = None

            if self._stop_event.is_set():
                break

            delay = self._compute_backoff_delay()
            logger.warning("Reconnecting in %.2f seconds", delay)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                continue

    async def _send_all_subscriptions(self) -> None:
        if self._ws is None:
            return
        for subscription in self._subscriptions:
            await self._ws.send(json.dumps(subscription.to_message()))

    def _compute_backoff_delay(self) -> float:
        jitter = random.uniform(0.8, 1.2)
        delay = min(self._backoff_seconds * jitter, BACKOFF_MAX_SECONDS)
        self._backoff_seconds = min(self._backoff_seconds * BACKOFF_MULTIPLIER, BACKOFF_MAX_SECONDS)
        return delay

    async def _handle_message(self, raw_message: str | bytes) -> None:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode()

        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning("Invalid websocket JSON: %s", raw_message[:200])
            return

        channel = message.get("channel")
        if channel == "pong":
            return
        if channel == "subscriptionResponse":
            logger.debug("Subscription confirmed: %s", message.get("data"))
            return
        if channel == "error":
            logger.warning("WebSocket error channel: %s", message.get("data"))
            raise ConnectionError(f"WebSocket error: {message.get('data')}")

        data = message.get("data")
        if data is None:
            return

        if channel == "trades":
            await self._enqueue_trades(data)
            return
        if channel in {"activeAssetCtx", "activeSpotAssetCtx"}:
            await self._enqueue_asset_ctx(data)
            return
        if channel == "webData2":
            await self._enqueue_web_data(data)

    async def _enqueue_trades(self, trades: list[dict[str, Any]]) -> None:
        for trade in trades:
            coin = str(trade.get("coin", ""))
            ts_ms = trade.get("time")
            ts = (
                datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if isinstance(ts_ms, (int, float))
                else datetime.now(timezone.utc)
            )
            payload = TradePayload(
                px=Decimal(str(trade.get("px", "0"))),
                sz=Decimal(str(trade.get("sz", "0"))),
                side=str(trade.get("side", "")),
                hash=str(trade.get("hash", "")),
                tid=trade.get("tid") if isinstance(trade.get("tid"), int) else None,
            )
            event = MarketEvent(kind=EventKind.TRADE, coin=coin, ts=ts, payload=payload)
            await self._put_event(event)

    async def _enqueue_asset_ctx(self, data: dict[str, Any]) -> None:
        coin = str(data.get("coin", ""))
        ctx = data.get("ctx") or {}
        ts = datetime.now(timezone.utc)
        payload = AssetCtxPayload(
            mark_px=Decimal(str(ctx.get("markPx", "0"))),
            open_interest=Decimal(str(ctx.get("openInterest", "0"))),
            funding=Decimal(str(ctx["funding"])) if ctx.get("funding") is not None else None,
            oracle_px=Decimal(str(ctx["oraclePx"])) if ctx.get("oraclePx") is not None else None,
            day_ntl_vlm=Decimal(str(ctx["dayNtlVlm"])) if ctx.get("dayNtlVlm") is not None else None,
        )
        event = MarketEvent(kind=EventKind.ASSET_CTX, coin=coin, ts=ts, payload=payload)
        await self._put_event(event)

        if self._on_tick is not None:
            await self._on_tick(
                MarketTick(
                    coin=coin,
                    mark_px=payload.mark_px,
                    funding=payload.funding,
                    ts=ts,
                )
            )

    async def _enqueue_web_data(self, data: dict[str, Any]) -> None:
        event = MarketEvent(
            kind=EventKind.WEB_DATA,
            coin=None,
            ts=datetime.now(timezone.utc),
            payload=data,
        )
        await self._put_event(event)

    async def _put_event(self, event: MarketEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await self._queue.put(event)
