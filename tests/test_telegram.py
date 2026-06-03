from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from src.core.config import HyperliquidSettings
from src.core.telegram import TelegramNotifier
from src.strategy.momentum_oi import SignalSide


@pytest.fixture
def dry_run_settings() -> HyperliquidSettings:
    return HyperliquidSettings.from_env().model_copy(
        update={
            "bot_dry_run": True,
            "telegram_bot_token": SecretStr("test-token"),
            "telegram_chat_id": "123456",
        }
    )


@pytest.fixture
def disabled_settings() -> HyperliquidSettings:
    return HyperliquidSettings.from_env().model_copy(
        update={
            "telegram_bot_token": SecretStr(""),
            "telegram_chat_id": "",
        }
    )


def test_notify_disabled_when_credentials_missing(disabled_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(disabled_settings)

    with patch("src.core.telegram.asyncio.create_task") as create_task:
        notifier.notify("hello")

    create_task.assert_not_called()
    assert notifier.enabled is False


@pytest.mark.asyncio
async def test_notify_entry_includes_dry_run_prefix(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_entry("BTC", SignalSide.LONG, Decimal("73600"), Decimal("0.05"))
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "DRY RUN / SANAL İŞLEM" in sent_text[0]
    assert "LONG Açıldı" in sent_text[0]
    assert "BTC" in sent_text[0]
    assert "$73600" in sent_text[0]
    assert "Sanal Adet" in sent_text[0]


@pytest.mark.asyncio
async def test_notify_exit_includes_roe_and_exit_reason(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_exit(
                "AVAX",
                Decimal("8.8577"),
                Decimal("1.57"),
                roe_pct=Decimal("15.7"),
                exit_reason="trailing_stop",
                side=SignalSide.LONG,
            )
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "AVAX" in sent_text[0]
    assert "+$1.57" in sent_text[0]
    assert "+%15.7 ROE" in sent_text[0]
    assert "Sebep: Trailing Stop" in sent_text[0]


@pytest.mark.asyncio
async def test_notify_uses_fire_and_forget(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    send_mock = AsyncMock()

    with patch.object(notifier, "_send", send_mock):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            notifier.notify("test message")
            create_task.assert_called_once()
            coro = create_task.call_args.args[0]
            await coro

    send_mock.assert_awaited_once_with("test message")


@pytest.mark.asyncio
async def test_notify_heartbeat_includes_symbols_and_period_stats(
    dry_run_settings: HyperliquidSettings,
) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_heartbeat(
                ["LINK", "SEI"],
                win_rate_pct=Decimal("46"),
                net_pnl_usd=Decimal("0.39"),
                trade_count=13,
                lookback_days=7,
            )
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "DRY RUN" in sent_text[0]
    assert "HEARTBEAT" in sent_text[0]
    assert "Açık Pozisyonlar: 2 (LINK, SEI)" in sent_text[0]
    assert "7 Günlük Win Rate: %46" in sent_text[0]
    assert "Net PnL: +$0.39" in sent_text[0]


@pytest.mark.asyncio
async def test_notify_heartbeat_without_closed_trades(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_heartbeat(
                [],
                win_rate_pct=Decimal("0"),
                net_pnl_usd=Decimal("0"),
                trade_count=0,
            )
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "Açık Pozisyonlar: 0" in sent_text[0]
    assert "7 Günlük: henüz kapalı işlem yok" in sent_text[0]


@pytest.mark.asyncio
async def test_notify_startup_includes_mode_and_recovery(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_startup(symbol_count=11, recovered=["LINK", "SEI"], journal_trade_count=13)
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "HL Futures Bot Başladı" in sent_text[0]
    assert "DRY RUN" in sent_text[0]
    assert "Coin: 11" in sent_text[0]
    assert "LINK, SEI" in sent_text[0]
    assert "/stats" in sent_text[0]


@pytest.mark.asyncio
async def test_notify_shutdown(dry_run_settings: HyperliquidSettings) -> None:
    notifier = TelegramNotifier(dry_run_settings)
    sent_text: list[str] = []

    async def capture_send(text: str) -> None:
        sent_text.append(text)

    with patch.object(notifier, "_send", side_effect=capture_send):
        with patch("src.core.telegram.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: __import__("asyncio").ensure_future(coro)
            notifier.notify_shutdown()
            await __import__("asyncio").sleep(0)

    assert len(sent_text) == 1
    assert "Bot kapatılıyor" in sent_text[0]
