"""Live Hyperliquid connection sanity check for REST meta and WebSocket market data."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import suppress

from src.core.config import HyperliquidSettings
from src.core.logger import setup_logging
from src.exchange.hyperliquid_rest import HyperliquidRestClient
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, HyperliquidWebSocketListener

logger = logging.getLogger(__name__)

LISTEN_SECONDS = 10.0
BTC_SYMBOL = "BTC"


async def main() -> int:
    settings = HyperliquidSettings.from_env()
    setup_logging(settings.log_level)

    btc_settings = settings.model_copy(update={"symbols": (BTC_SYMBOL,)})
    rest = HyperliquidRestClient(settings)
    listener = HyperliquidWebSocketListener(btc_settings)
    ws_task: asyncio.Task[None] | None = None
    saw_btc_ctx = False

    try:
        await rest.initialize()
        btc_index = rest._coin_to_asset[BTC_SYMBOL]
        logger.info("BTC asset_index=%s", btc_index)

        await listener.subscribe_trades(BTC_SYMBOL)
        ws_task = asyncio.create_task(listener.run())

        deadline = time.monotonic() + LISTEN_SECONDS
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                event = await asyncio.wait_for(listener.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            if event.kind != EventKind.ASSET_CTX or event.coin != BTC_SYMBOL:
                continue
            if not isinstance(event.payload, AssetCtxPayload):
                continue

            saw_btc_ctx = True
            print(
                f"[{BTC_SYMBOL}] mark_px={event.payload.mark_px}  "
                f"open_interest={event.payload.open_interest}  "
                f"ts={event.ts.isoformat()}"
            )

        if not saw_btc_ctx:
            logger.warning("No BTC activeAssetCtx events received within %.0f seconds", LISTEN_SECONDS)
            return 1
        return 0
    finally:
        await listener.disconnect()
        if ws_task is not None:
            ws_task.cancel()
            with suppress(asyncio.CancelledError):
                await ws_task
        await rest.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
