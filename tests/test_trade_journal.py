from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.core.config import HyperliquidSettings
from src.core.trade_journal import TradeJournal, calc_roe_pct
from src.execution.position_manager import ManagedPosition
from src.strategy.momentum_oi import SignalSide


@pytest.fixture
def journal_path(tmp_path: Path) -> Path:
    return tmp_path / "trades.jsonl"


@pytest.fixture
def journal(journal_path: Path) -> TradeJournal:
    settings = HyperliquidSettings.from_env().model_copy(
        update={"trade_journal_path": str(journal_path), "bot_dry_run": True, "leverage": 10}
    )
    return TradeJournal(settings, path=journal_path)


def _position(
    *,
    side: SignalSide = SignalSide.LONG,
    entry_px: str = "100",
    size: str = "1",
    opened_at: datetime | None = None,
    coin: str = "AVAX",
) -> ManagedPosition:
    return ManagedPosition(
        coin=coin,
        side=side,
        entry_px=Decimal(entry_px),
        size=Decimal(size),
        stop_px=Decimal("98"),
        opened_at=opened_at or datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


def test_calc_roe_pct_long_with_leverage() -> None:
    roe = calc_roe_pct(
        side=SignalSide.LONG,
        entry_px=Decimal("8.7206"),
        exit_px=Decimal("8.8577"),
        leverage=10,
    )
    assert roe.quantize(Decimal("0.1")) == Decimal("15.7")


def test_calc_roe_pct_short_with_leverage() -> None:
    roe = calc_roe_pct(
        side=SignalSide.SHORT,
        entry_px=Decimal("100"),
        exit_px=Decimal("99"),
        leverage=10,
    )
    assert roe == Decimal("10")


def test_record_and_load_closed_trade(journal: TradeJournal, journal_path: Path) -> None:
    position = _position()
    closed_at = datetime(2026, 6, 1, 12, 45, tzinfo=timezone.utc)
    journal.record_closed_trade(
        position,
        Decimal("101.57"),
        Decimal("1.57"),
        Decimal("15.7"),
        "trailing_stop",
        closed_at=closed_at,
    )

    assert journal_path.exists()
    trades = journal.load_closed_trades(since=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert len(trades) == 1
    assert trades[0].symbol == "AVAX"
    assert trades[0].exit_reason == "trailing_stop"
    assert trades[0].pnl_usd == Decimal("1.57")
    assert trades[0].dry_run is True


def test_period_stats_win_rate_and_net_pnl(journal: TradeJournal) -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    journal.record_closed_trade(
        _position(),
        Decimal("101"),
        Decimal("1"),
        Decimal("10"),
        "trailing_stop",
        closed_at=now - timedelta(days=1),
    )
    journal.record_closed_trade(
        _position(side=SignalSide.SHORT),
        Decimal("99"),
        Decimal("-0.5"),
        Decimal("-5"),
        "stop_loss",
        closed_at=now - timedelta(days=2),
    )
    journal.record_closed_trade(
        _position(),
        Decimal("100"),
        Decimal("0.5"),
        Decimal("5"),
        "trailing_stop",
        closed_at=now - timedelta(days=10),
    )

    stats = journal.period_stats(days=7, now=now)
    assert stats.trade_count == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate_pct == Decimal("50.0")
    assert stats.net_pnl_usd == Decimal("0.50")


def test_load_closed_trades_skips_malformed_lines(journal: TradeJournal, journal_path: Path) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    valid = {
        "symbol": "BTC",
        "side": "long",
        "entry_px": "100",
        "exit_px": "101",
        "size": "1",
        "pnl_usd": "1",
        "roe_pct": "10",
        "exit_reason": "trailing_stop",
        "opened_at": "2026-06-01T12:00:00+00:00",
        "closed_at": "2026-06-01T13:00:00+00:00",
        "dry_run": True,
    }
    journal_path.write_text(
        "not-json\n" + json.dumps(valid) + "\n{broken",
        encoding="utf-8",
    )

    trades = journal.load_closed_trades(since=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert len(trades) == 1
    assert trades[0].symbol == "BTC"


def test_load_all_closed_trades(journal: TradeJournal) -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    journal.record_closed_trade(
        _position(),
        Decimal("101"),
        Decimal("1"),
        Decimal("10"),
        "trailing_stop",
        closed_at=now - timedelta(days=30),
    )
    journal.record_closed_trade(
        _position(coin="ETH"),
        Decimal("102"),
        Decimal("2"),
        Decimal("20"),
        "trailing_stop",
        closed_at=now - timedelta(days=1),
    )
    assert len(journal.load_all_closed_trades()) == 2
