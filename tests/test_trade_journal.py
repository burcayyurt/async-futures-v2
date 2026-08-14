from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.core.config import HyperliquidSettings
from src.core.config_fingerprint import config_id, register_config
from src.core.trade_journal import TradeJournal, calc_fee_usd, calc_roe_pct
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


@pytest.mark.asyncio
async def test_record_and_load_closed_trade(journal: TradeJournal, journal_path: Path) -> None:
    position = _position()
    closed_at = datetime(2026, 6, 1, 12, 45, tzinfo=timezone.utc)
    await journal.record_closed_trade(
        position,
        Decimal("101.57"),
        Decimal("1.57"),
        Decimal("15.7"),
        "trailing_stop",
        closed_at=closed_at,
    )

    assert journal_path.exists()
    trades = await journal.load_closed_trades(since=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert len(trades) == 1
    assert trades[0].symbol == "AVAX"
    assert trades[0].exit_reason == "trailing_stop"
    assert trades[0].pnl_usd == Decimal("1.57")
    assert trades[0].dry_run is True


@pytest.mark.asyncio
async def test_period_stats_win_rate_and_net_pnl(journal: TradeJournal) -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    await journal.record_closed_trade(
        _position(),
        Decimal("101"),
        Decimal("1"),
        Decimal("10"),
        "trailing_stop",
        closed_at=now - timedelta(days=1),
    )
    await journal.record_closed_trade(
        _position(side=SignalSide.SHORT),
        Decimal("99"),
        Decimal("-0.5"),
        Decimal("-5"),
        "stop_loss",
        closed_at=now - timedelta(days=2),
    )
    await journal.record_closed_trade(
        _position(),
        Decimal("100"),
        Decimal("0.5"),
        Decimal("5"),
        "trailing_stop",
        closed_at=now - timedelta(days=10),
    )

    stats = await journal.period_stats(days=7, now=now)
    assert stats.trade_count == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate_pct == Decimal("50.0")
    # Net of fees, not gross: +1.00 and -0.50 gross cost 0.06045 and 0.05955 in
    # round-trip fees (maker entry 1.5bps, taker exit 4.5bps), leaving 0.38.
    assert stats.net_pnl_usd == Decimal("0.38")


def test_config_id_changes_with_outcome_relevant_settings() -> None:
    """Fingerprint must move when a parameter that changes trades changes."""
    base = HyperliquidSettings.from_env()
    same = base.model_copy(update={"log_level": "DEBUG"})  # cosmetic
    different = base.model_copy(update={"trailing_callback_pct": Decimal("0.02")})

    assert config_id(same) == config_id(base)
    assert config_id(different) != config_id(base)


def test_config_id_ignores_decimal_formatting() -> None:
    """0.35 and 0.3500 are the same configuration, not two."""
    base = HyperliquidSettings.from_env()
    a = base.model_copy(update={"trailing_callback_pct": Decimal("0.0035")})
    b = base.model_copy(update={"trailing_callback_pct": Decimal("0.00350")})
    assert config_id(a) == config_id(b)


def test_register_config_appends_once_per_configuration(tmp_path: Path) -> None:
    registry = tmp_path / "config_registry.jsonl"
    settings = HyperliquidSettings.from_env()

    first = register_config(settings, registry)
    second = register_config(settings, registry)  # same config, must not duplicate
    other = register_config(
        settings.model_copy(update={"trailing_callback_pct": Decimal("0.02")}), registry
    )

    assert first == second
    assert other != first
    lines = [line for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    recorded = {json.loads(line)["config_id"] for line in lines}
    assert recorded == {first, other}
    # The snapshot must be readable enough to reconstruct what ran.
    snap = json.loads(lines[0])["settings"]
    assert "trailing_callback_pct" in snap and "symbols" in snap


@pytest.mark.asyncio
async def test_recorded_trade_carries_config_id(journal: TradeJournal) -> None:
    await journal.record_closed_trade(
        _position(), Decimal("101"), Decimal("1"), Decimal("10"), "trailing_stop"
    )
    trade = (await journal.load_all_closed_trades())[0]
    assert trade.config_id == config_id(HyperliquidSettings.from_env())


@pytest.mark.asyncio
async def test_legacy_rows_are_marked_not_guessed(
    journal: TradeJournal, journal_path: Path
) -> None:
    """Pre-fingerprint trades must be identifiable, never attributed by default."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        json.dumps(
            {
                "symbol": "BTC", "side": "long", "entry_px": "100", "exit_px": "101",
                "size": "1", "pnl_usd": "1", "roe_pct": "10",
                "exit_reason": "trailing_stop",
                "opened_at": "2026-06-01T12:00:00+00:00",
                "closed_at": "2026-06-01T13:00:00+00:00", "dry_run": True,
            }
        ) + "\n",
        encoding="utf-8",
    )
    trade = (await journal.load_all_closed_trades())[0]
    assert trade.config_id == "legacy"


def test_calc_fee_usd_charges_taker_on_exit_always() -> None:
    maker_in = calc_fee_usd(
        entry_px=Decimal("100"),
        exit_px=Decimal("100"),
        size=Decimal("1"),
        maker_entry=True,
        maker_fee_bps=Decimal("1.5"),
        taker_fee_bps=Decimal("4.5"),
    )
    taker_in = calc_fee_usd(
        entry_px=Decimal("100"),
        exit_px=Decimal("100"),
        size=Decimal("1"),
        maker_entry=False,
        maker_fee_bps=Decimal("1.5"),
        taker_fee_bps=Decimal("4.5"),
    )
    # Exit is IOC reduce-only in every path, so only the entry leg changes rate.
    assert maker_in == Decimal("0.015") + Decimal("0.045")
    assert taker_in == Decimal("0.045") + Decimal("0.045")


@pytest.mark.asyncio
async def test_record_closed_trade_persists_fee_and_net(journal: TradeJournal) -> None:
    await journal.record_closed_trade(
        _position(),
        Decimal("101"),
        Decimal("1"),
        Decimal("10"),
        "trailing_stop",
    )
    trade = (await journal.load_all_closed_trades())[0]
    # Fee is 0.015 (maker entry) + 0.04545 (taker exit) = 0.06045, stored at
    # 4dp under Decimal's default banker's rounding.
    assert trade.pnl_usd == Decimal("1")
    assert trade.fee_usd == Decimal("0.0604")
    assert trade.net_pnl_usd == Decimal("0.9396")
    assert trade.fee_usd + trade.net_pnl_usd == trade.pnl_usd


@pytest.mark.asyncio
async def test_legacy_records_without_fee_fields_stay_readable(
    journal: TradeJournal, journal_path: Path
) -> None:
    """Pre-fee-accounting rows carry no fee fields; net falls back to gross."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
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
    journal_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    trade = (await journal.load_all_closed_trades())[0]
    assert trade.fee_usd == Decimal("0")
    assert trade.net_pnl_usd == Decimal("1")


@pytest.mark.asyncio
async def test_load_closed_trades_skips_malformed_lines(journal: TradeJournal, journal_path: Path) -> None:
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

    trades = await journal.load_closed_trades(since=datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert len(trades) == 1
    assert trades[0].symbol == "BTC"


@pytest.mark.asyncio
async def test_load_all_closed_trades(journal: TradeJournal) -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    await journal.record_closed_trade(
        _position(),
        Decimal("101"),
        Decimal("1"),
        Decimal("10"),
        "trailing_stop",
        closed_at=now - timedelta(days=30),
    )
    await journal.record_closed_trade(
        _position(coin="ETH"),
        Decimal("102"),
        Decimal("2"),
        Decimal("20"),
        "trailing_stop",
        closed_at=now - timedelta(days=1),
    )
    assert len(await journal.load_all_closed_trades()) == 2


@pytest.mark.asyncio
async def test_concurrent_record_writes_are_serialized(journal: TradeJournal) -> None:
    """Multiple concurrent record_closed_trade calls should not lose data."""
    import asyncio
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    tasks = [
        journal.record_closed_trade(
            _position(coin=f"C{i}"),
            Decimal("101"),
            Decimal("1"),
            Decimal("10"),
            "trailing_stop",
            closed_at=now,
        )
        for i in range(10)
    ]
    await asyncio.gather(*tasks)
    trades = await journal.load_all_closed_trades()
    assert len(trades) == 10
