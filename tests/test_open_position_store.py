from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.core.config import HyperliquidSettings
from src.core.open_position_store import OpenPositionStore
from src.execution.position_manager import ManagedPosition
from src.strategy.momentum_oi import SignalSide


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "open_positions.json"


@pytest.fixture
def store(store_path: Path) -> OpenPositionStore:
    settings = HyperliquidSettings.from_env().model_copy(
        update={"open_positions_path": str(store_path), "bot_dry_run": True}
    )
    return OpenPositionStore(settings, path=store_path)


def _position(coin: str = "LINK") -> ManagedPosition:
    return ManagedPosition(
        coin=coin,
        side=SignalSide.LONG,
        entry_px=Decimal("9.09"),
        size=Decimal("10"),
        stop_px=Decimal("8.91"),
        peak_px=Decimal("9.20"),
        opened_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_load_creates_empty_file(store: OpenPositionStore, store_path: Path) -> None:
    loaded = await store.load()
    assert loaded == {}
    assert store_path.exists()


@pytest.mark.asyncio
async def test_upsert_and_load_round_trip(store: OpenPositionStore) -> None:
    position = _position()
    await store.upsert(position)
    loaded = await store.load()
    assert "LINK" in loaded
    assert loaded["LINK"].entry_px == Decimal("9.09")
    assert loaded["LINK"].peak_px == Decimal("9.20")


@pytest.mark.asyncio
async def test_remove_deletes_position(store: OpenPositionStore) -> None:
    await store.upsert(_position())
    await store.remove("LINK")
    loaded = await store.load()
    assert "LINK" not in loaded
