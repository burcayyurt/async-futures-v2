from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from src.core.trade_journal import PeriodStats
from src.execution.position_manager import ManagedPosition
from src.strategy.momentum_oi import SignalSide
from src.telegram.messages import format_help_message, format_positions_message, format_stats_message


def test_format_stats_message_with_trades() -> None:
    stats = PeriodStats(
        trade_count=13,
        wins=6,
        losses=7,
        win_rate_pct=Decimal("46.2"),
        net_pnl_usd=Decimal("0.39"),
        lookback_days=7,
    )
    text = format_stats_message(stats, all_time_count=20)
    assert "İstatistikler (7 Gün)" in text
    assert "Win Rate: %46.2" in text
    assert "Toplam kayıtlı kapalı işlem: 20" in text


def test_format_positions_message_lists_open_positions() -> None:
    now = datetime(2026, 6, 1, 14, 15, tzinfo=timezone.utc)
    positions = [
        ManagedPosition(
            coin="LINK",
            side=SignalSide.LONG,
            entry_px=Decimal("9.09"),
            size=Decimal("10"),
            stop_px=Decimal("8.91"),
            opened_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        )
    ]
    text = format_positions_message(positions, now=now)
    assert "Açık Pozisyonlar (1)" in text
    assert "LINK LONG" in text
    assert "$9.09" in text


def test_format_help_message_lists_commands() -> None:
    text = format_help_message()
    assert "/stats" in text
    assert "/positions" in text
    assert "/help" in text
