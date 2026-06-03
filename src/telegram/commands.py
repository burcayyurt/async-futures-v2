from __future__ import annotations

import logging
from enum import Enum

import aiohttp

from src.core.config import HyperliquidSettings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramCommandType(str, Enum):
    STATS = "stats"
    POSITIONS = "positions"
    HELP = "help"


class ParsedTelegramCommand:
    def __init__(self, command: TelegramCommandType, chat_id: str, raw_text: str) -> None:
        self.command = command
        self.chat_id = chat_id
        self.raw_text = raw_text


_STATS_ALIASES = {"/stats", "/performance", "stats"}
_POSITIONS_ALIASES = {"/positions", "positions"}
_HELP_ALIASES = {"/help", "help"}


def parse_command_text(text: str) -> TelegramCommandType | None:
    normalized = text.strip().lower()
    if not normalized:
        return None
    first_token = normalized.split()[0]
    if first_token in _STATS_ALIASES:
        return TelegramCommandType.STATS
    if first_token in _POSITIONS_ALIASES:
        return TelegramCommandType.POSITIONS
    if first_token in _HELP_ALIASES:
        return TelegramCommandType.HELP
    return None


class TelegramCommandPoller:
    def __init__(self, settings: HyperliquidSettings) -> None:
        self._settings = settings
        self._offset: int | None = None
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def poll_commands(self) -> list[ParsedTelegramCommand]:
        token = self._settings.telegram_bot_token.get_secret_value().strip()
        if not token:
            return []

        url = f"{TELEGRAM_API_BASE}/bot{token}/getUpdates"
        params: dict[str, str | int] = {"timeout": 0}
        if self._offset is not None:
            params["offset"] = self._offset

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                payload = await response.json(content_type=None)
        except Exception:
            logger.exception("Telegram poll_commands request failed")
            return []

        if payload.get("ok") is not True:
            return []

        commands: list[ParsedTelegramCommand] = []
        allowed_chat_id = self._settings.telegram_chat_id.strip()
        for update in payload.get("result", []):
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            text = str(message.get("text", "")).strip()
            chat_id = str(chat.get("id", ""))
            if allowed_chat_id and chat_id != allowed_chat_id:
                continue
            command_type = parse_command_text(text)
            if command_type is not None:
                commands.append(ParsedTelegramCommand(command_type, chat_id, text))
        return commands

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
