"""Async event-driven orchestrator for the Hyperliquid perpetual futures bot."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress
from decimal import Decimal
from typing import Any

from src.core.config import HyperliquidSettings
from src.core.logger import setup_logging
from src.core.open_position_store import OpenPositionStore
from src.core.telegram import TelegramNotifier
from src.telegram.messages import format_help_message, format_positions_message, format_stats_message
from src.core.trade_journal import TradeJournal
from src.exchange.hyperliquid_rest import HyperliquidRestClient
from src.exchange.hyperliquid_ws import AssetCtxPayload, EventKind, HyperliquidWebSocketListener, MarketEvent
from src.execution.order_router import OrderRouter
from src.execution.position_manager import PositionManager
from src.risk.kill_switch import KillSwitch
from src.risk.margin_manager import MarginManager
from src.strategy.momentum_oi import MomentumOIStrategy, TradeSignal
from src.telegram.commands import TelegramCommandPoller, TelegramCommandType

logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = 10_000
DEFAULT_REGIME_COIN = "BTC"
HEARTBEAT_INTERVAL_SECONDS = 3600
ACTIVITY_LOG_INTERVAL_SECONDS = 300


def _parse_equity_from_web_data(payload: dict[str, Any]) -> Decimal | None:
    clearinghouse = payload.get("clearinghouseState")
    if not isinstance(clearinghouse, dict):
        return None
    margin_summary = clearinghouse.get("marginSummary")
    if not isinstance(margin_summary, dict):
        return None
    account_value = margin_summary.get("accountValue")
    if account_value is None:
        return None
    return Decimal(str(account_value))


def _parse_regime_mark_from_web_data(payload: dict[str, Any], coin: str) -> Decimal | None:
    asset_ctxs = payload.get("assetCtxs")
    meta = payload.get("meta")
    if not isinstance(asset_ctxs, list) or not isinstance(meta, dict):
        return None

    universe = meta.get("universe")
    if not isinstance(universe, list):
        return None

    normalized = coin.strip().upper()
    asset_index: int | None = None
    for index, entry in enumerate(universe):
        if isinstance(entry, dict) and str(entry.get("name", "")).upper() == normalized:
            asset_index = index
            break
    if asset_index is None or asset_index >= len(asset_ctxs):
        return None

    ctx = asset_ctxs[asset_index]
    if not isinstance(ctx, dict):
        return None
    mark_px = ctx.get("markPx")
    if mark_px is None:
        return None
    return Decimal(str(mark_px))


async def market_fanout(
    source: asyncio.Queue[MarketEvent],
    strategy_q: asyncio.Queue[MarketEvent],
    position_q: asyncio.Queue[MarketEvent],
    risk_q: asyncio.Queue[MarketEvent],
) -> None:
    while True:
        event = await source.get()
        if event.kind in {EventKind.TRADE, EventKind.ASSET_CTX}:
            await strategy_q.put(event)
        if event.kind == EventKind.ASSET_CTX:
            await position_q.put(event)
        if event.kind in {EventKind.WEB_DATA, EventKind.ASSET_CTX}:
            await risk_q.put(event)


async def risk_worker(
    queue: asyncio.Queue[MarketEvent],
    kill_switch: KillSwitch,
    *,
    regime_coin: str = DEFAULT_REGIME_COIN,
    wallet_configured: bool = True,
) -> None:
    if not wallet_configured:
        logger.warning(
            "WALLET_ADDRESS not set; webData2 drawdown monitoring disabled (dry-run OK)"
        )

    latest_equity: Decimal | None = None
    latest_mark: Decimal | None = None
    normalized_regime = regime_coin.strip().upper()

    async def recheck_equity(_mark_price: Decimal) -> Decimal:
        return latest_equity or Decimal("0")

    while True:
        event = await queue.get()
        try:
            if event.kind == EventKind.ASSET_CTX and event.coin == normalized_regime:
                if isinstance(event.payload, AssetCtxPayload):
                    latest_mark = event.payload.mark_px
            elif event.kind == EventKind.WEB_DATA:
                if not isinstance(event.payload, dict):
                    continue
                equity = _parse_equity_from_web_data(event.payload)
                if equity is not None:
                    latest_equity = equity
                mark = _parse_regime_mark_from_web_data(event.payload, normalized_regime)
                if mark is not None:
                    latest_mark = mark
                if latest_equity is not None and latest_mark is not None:
                    await kill_switch.check_drawdown(
                        latest_equity,
                        latest_mark,
                        coin=normalized_regime,
                        recheck_equity=recheck_equity,
                    )
        except Exception:
            logger.exception("Risk worker failed processing event")


async def heartbeat_worker(
    telegram: TelegramNotifier,
    position_manager: PositionManager,
    trade_journal: TradeJournal,
    settings: HyperliquidSettings,
    *,
    interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
    lookback_days: int = 7,
) -> None:
    await asyncio.sleep(interval_seconds)
    while True:
        open_symbols = sorted(position_manager.positions.keys())
        stats = await trade_journal.period_stats(days=lookback_days)
        if open_symbols:
            positions_line = f"Açık Pozisyonlar: {len(open_symbols)} ({', '.join(open_symbols)})"
        else:
            positions_line = "Açık Pozisyonlar: 0"

        if stats.trade_count == 0:
            stats_line = f"{lookback_days} Günlük: henüz kapalı işlem yok"
        else:
            stats_line = (
                f"{lookback_days} Günlük Win Rate: %{stats.win_rate_pct} | "
                f"Net PnL: {stats.net_pnl_usd:+.2f}"
            )

        log_line = f"💓 [HEARTBEAT] Bot aktif olarak piyasayı dinliyor. {positions_line} | {stats_line}"
        if settings.bot_dry_run:
            logger.info("🧪 [DRY RUN] %s", log_line)
        else:
            logger.info(log_line)

        telegram.notify_heartbeat(
            open_symbols,
            win_rate_pct=stats.win_rate_pct,
            net_pnl_usd=stats.net_pnl_usd,
            trade_count=stats.trade_count,
            lookback_days=lookback_days,
        )
        await asyncio.sleep(interval_seconds)


async def activity_worker(
    strategy: MomentumOIStrategy,
    position_manager: PositionManager,
    settings: HyperliquidSettings,
    *,
    interval_seconds: int = ACTIVITY_LOG_INTERVAL_SECONDS,
) -> None:
    await asyncio.sleep(interval_seconds)
    while True:
        open_positions = list(position_manager.positions.keys())
        logger.info(
            "Activity summary | open=%s | %s",
            ",".join(open_positions) if open_positions else "none",
            strategy.decision_summary(settings.symbols),
        )
        await asyncio.sleep(interval_seconds)


async def telegram_command_worker(
    settings: HyperliquidSettings,
    telegram: TelegramNotifier,
    trade_journal: TradeJournal,
    position_manager: PositionManager,
    poller: TelegramCommandPoller,
) -> None:
    while True:
        try:
            commands = await poller.poll_commands()
            for command in commands:
                if command.command == TelegramCommandType.STATS:
                    stats = await trade_journal.period_stats(days=7)
                    all_time_count = len(await trade_journal.load_all_closed_trades())
                    await telegram.send_message(format_stats_message(stats, all_time_count=all_time_count))
                elif command.command == TelegramCommandType.POSITIONS:
                    positions = list(position_manager.positions.values())
                    await telegram.send_message(format_positions_message(positions))
                elif command.command == TelegramCommandType.HELP:
                    await telegram.send_message(format_help_message())
        except Exception:
            logger.exception("Telegram command polling failed")
        await asyncio.sleep(settings.telegram_poll_interval_seconds)


async def shutdown(
    tasks: list[asyncio.Task[Any]],
    ws: HyperliquidWebSocketListener,
    rest: HyperliquidRestClient,
    telegram: TelegramNotifier,
    command_poller: TelegramCommandPoller | None = None,
) -> None:
    telegram.notify_shutdown()
    await ws.disconnect()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if command_poller is not None:
        await command_poller.close()
    await rest.close()
    await telegram.close()


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    def _request_shutdown() -> None:
        logger.info("Shutdown requested")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_shutdown)


async def run_bot() -> int:
    settings = HyperliquidSettings.from_env()
    setup_logging(settings.log_level)

    if not settings.bot_dry_run:
        settings.require_credentials()

    wallet_configured = bool(settings.wallet_address.strip())
    regime_coin = DEFAULT_REGIME_COIN

    rest = HyperliquidRestClient(settings)
    await rest.initialize()

    ws = HyperliquidWebSocketListener(settings)
    telegram = TelegramNotifier(settings)
    trade_journal = TradeJournal(settings)
    open_position_store = OpenPositionStore(settings)
    kill_switch = KillSwitch(settings, telegram=telegram)
    margin_manager = MarginManager(settings, rest)
    position_manager = PositionManager(
        settings,
        kill_switch,
        telegram=telegram,
        trade_journal=trade_journal,
        open_position_store=open_position_store,
    )
    order_router = OrderRouter(
        settings,
        rest,
        kill_switch,
        margin_manager,
        position_manager,
        telegram=telegram,
    )

    signal_queue: asyncio.Queue[TradeSignal] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    strategy_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    position_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    risk_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    strategy = MomentumOIStrategy(
        settings,
        on_signal=lambda sig: signal_queue.put_nowait(sig),
    )

    recovered = await position_manager.recover_positions()
    recovered_symbols = sorted(position.coin for position in recovered)
    all_trades = await trade_journal.load_all_closed_trades()
    journal_trade_count = len(all_trades)
    telegram.notify_startup(
        symbol_count=len(settings.symbols),
        recovered=recovered_symbols,
        journal_trade_count=journal_trade_count,
    )
    if recovered_symbols:
        logger.info("Recovered open positions on startup: %s", ",".join(recovered_symbols))

    logger.info(
        "Bot starting symbols=%s dry_run=%s leverage=%s margin_mode=%s telegram=%s",
        ",".join(settings.symbols),
        settings.bot_dry_run,
        settings.leverage,
        settings.margin_mode,
        "enabled" if telegram.enabled else "disabled",
    )
    logger.info(
        "Strategy params price_delta=%s%% oi_min_increase=%s%% volume_spike=%sx "
        "window=%ss min_trades=%s ema=%s regime=%s buffer=%s%%",
        settings.strategy_price_delta_pct,
        settings.strategy_min_oi_increase_pct,
        settings.strategy_volume_spike_multiplier,
        settings.strategy_window_seconds,
        settings.strategy_min_trades_in_window,
        settings.strategy_ema_period,
        settings.strategy_regime_coin,
        settings.strategy_regime_buffer_pct,
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, shutdown_event)

    tasks: list[asyncio.Task[Any]] = []
    command_poller: TelegramCommandPoller | None = None

    try:
        tasks = [
            asyncio.create_task(ws.run(), name="ws"),
            asyncio.create_task(
                market_fanout(ws.queue, strategy_queue, position_queue, risk_queue),
                name="market_fanout",
            ),
            asyncio.create_task(strategy.consume(strategy_queue), name="strategy"),
            asyncio.create_task(order_router.consume_signals(signal_queue), name="order_router"),
            asyncio.create_task(
                position_manager.consume_market_events(position_queue),
                name="position_manager",
            ),
            asyncio.create_task(
                risk_worker(
                    risk_queue,
                    kill_switch,
                    regime_coin=regime_coin,
                    wallet_configured=wallet_configured,
                ),
                name="risk_worker",
            ),
            asyncio.create_task(
                heartbeat_worker(telegram, position_manager, trade_journal, settings),
                name="heartbeat",
            ),
            asyncio.create_task(
                activity_worker(strategy, position_manager, settings),
                name="activity",
            ),
        ]
        if telegram.enabled and settings.telegram_poll_enabled:
            command_poller = TelegramCommandPoller(settings)
            tasks.append(
                asyncio.create_task(
                    telegram_command_worker(settings, telegram, trade_journal, position_manager, command_poller),
                    name="telegram_commands",
                )
            )
        await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")
    finally:
        await shutdown(tasks, ws, rest, telegram, command_poller=command_poller)

    logger.info("Bot stopped cleanly")
    return 0


def main() -> None:
    try:
        code = asyncio.run(run_bot())
    except KeyboardInterrupt:
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
