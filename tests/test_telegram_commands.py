from __future__ import annotations

import time

import pytest

from src.telegram.commands import TelegramCommandType, parse_command_text


def test_parse_stats_commands() -> None:
    assert parse_command_text("/stats") == TelegramCommandType.STATS
    assert parse_command_text("/performance") == TelegramCommandType.STATS
    assert parse_command_text("stats") == TelegramCommandType.STATS


def test_parse_positions_commands() -> None:
    assert parse_command_text("/positions") == TelegramCommandType.POSITIONS
    assert parse_command_text("positions") == TelegramCommandType.POSITIONS


def test_parse_help_commands() -> None:
    assert parse_command_text("/help") == TelegramCommandType.HELP


def test_parse_unknown_command_returns_none() -> None:
    assert parse_command_text("/kill") is None
    assert parse_command_text("hello") is None
    assert parse_command_text("") is None


# ------------------------------------------------- outage backoff behaviour


def _poller_with_token():
    from pydantic import SecretStr

    from src.core.config import HyperliquidSettings
    from src.telegram.commands import TelegramCommandPoller

    settings = HyperliquidSettings.from_env().model_copy(
        update={"telegram_bot_token": SecretStr("token"), "telegram_chat_id": "1"}
    )
    return TelegramCommandPoller(settings)


@pytest.mark.asyncio
async def test_unreachable_telegram_backs_off_instead_of_hammering(monkeypatch) -> None:
    """A dead endpoint must not be retried on every poll tick."""
    poller = _poller_with_token()
    attempts = 0

    async def _boom():
        nonlocal attempts
        attempts += 1
        raise ConnectionError("dns failure")

    monkeypatch.setattr(poller, "_get_session", _boom)

    assert await poller.poll_commands() == []
    assert attempts == 1

    # Subsequent polls inside the backoff window must not touch the network.
    assert await poller.poll_commands() == []
    assert await poller.poll_commands() == []
    assert attempts == 1


@pytest.mark.asyncio
async def test_backoff_grows_and_is_capped(monkeypatch) -> None:
    poller = _poller_with_token()

    async def _boom():
        raise ConnectionError("dns failure")

    monkeypatch.setattr(poller, "_get_session", _boom)

    delays = []
    for _ in range(12):
        poller._retry_at = 0.0  # force an attempt each iteration
        before = time.monotonic()
        await poller.poll_commands()
        delays.append(poller._retry_at - before)

    assert delays[0] < delays[1] < delays[2]  # grows
    assert max(delays) <= poller.BACKOFF_MAX_SECONDS + 1  # capped


@pytest.mark.asyncio
async def test_recovery_resets_the_backoff(monkeypatch) -> None:
    poller = _poller_with_token()

    async def _boom():
        raise ConnectionError("dns failure")

    monkeypatch.setattr(poller, "_get_session", _boom)
    await poller.poll_commands()
    assert poller._consecutive_failures == 1

    class _Resp:
        async def json(self, content_type=None):
            return {"ok": True, "result": []}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def get(self, *a, **k):
            return _Resp()

    async def _ok():
        return _Session()

    monkeypatch.setattr(poller, "_get_session", _ok)
    poller._retry_at = 0.0
    await poller.poll_commands()

    assert poller._consecutive_failures == 0
    assert poller._retry_at == 0.0
