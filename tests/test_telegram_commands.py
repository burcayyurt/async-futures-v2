from __future__ import annotations

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
