from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import suppress
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.core.config import HyperliquidSettings
from src.strategy.momentum_oi import SignalSide

if TYPE_CHECKING:
    from src.execution.position_manager import ManagedPosition

logger = logging.getLogger(__name__)


class StoredPosition(BaseModel):
    coin: str
    side: str
    entry_px: str
    size: str
    stop_px: str
    break_even_armed: bool = False
    peak_px: str | None = None
    opened_at: datetime
    dry_run: bool = True


class OpenPositionsFile(BaseModel):
    version: int = 1
    positions: list[StoredPosition] = Field(default_factory=list)


def stored_from_managed(position: ManagedPosition, *, dry_run: bool) -> StoredPosition:
    return StoredPosition(
        coin=position.coin.strip().upper(),
        side=position.side.value,
        entry_px=str(position.entry_px),
        size=str(position.size),
        stop_px=str(position.stop_px),
        break_even_armed=position.break_even_armed,
        peak_px=str(position.peak_px) if position.peak_px is not None else None,
        opened_at=position.opened_at,
        dry_run=dry_run,
    )


def managed_from_stored(stored: StoredPosition) -> ManagedPosition:
    from src.execution.position_manager import ManagedPosition

    side = SignalSide.LONG if stored.side.lower() == SignalSide.LONG.value else SignalSide.SHORT
    peak_px = Decimal(stored.peak_px) if stored.peak_px is not None else None
    return ManagedPosition(
        coin=stored.coin.strip().upper(),
        side=side,
        entry_px=Decimal(stored.entry_px),
        size=Decimal(stored.size),
        stop_px=Decimal(stored.stop_px),
        break_even_armed=stored.break_even_armed,
        peak_px=peak_px,
        opened_at=stored.opened_at,
    )


class OpenPositionStore:
    """Persists open positions to disk for crash recovery."""

    def __init__(self, settings: HyperliquidSettings, *, path: Path | None = None) -> None:
        self._settings = settings
        self._path = path or Path(settings.open_positions_path)
        self._lock = asyncio.Lock()
        self._positions: dict[str, ManagedPosition] = {}

    @property
    def path(self) -> Path:
        return self._path

    async def load(self) -> dict[str, ManagedPosition]:
        async with self._lock:
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                await self._write_file(OpenPositionsFile())
                logger.info("Created empty open positions file at %s", self._path)
                self._positions = {}
                return {}

            def _read() -> tuple[OpenPositionsFile, bool]:
                raw = self._path.read_text(encoding="utf-8")
                if not raw.strip():
                    return OpenPositionsFile(), True
                return OpenPositionsFile.model_validate_json(raw), False

            try:
                payload, rewrite_empty = await asyncio.to_thread(_read)
            except Exception:
                logger.warning("Failed to read open positions from %s", self._path, exc_info=True)
                self._positions = {}
                return {}

            loaded = {stored.coin.strip().upper(): managed_from_stored(stored) for stored in payload.positions}
            self._positions = loaded
            if rewrite_empty:
                await self._write_file(OpenPositionsFile())
            logger.info("Loaded %d open position(s) from %s", len(loaded), self._path)
            return dict(loaded)

    async def upsert(self, position: ManagedPosition) -> None:
        async with self._lock:
            coin = position.coin.strip().upper()
            self._positions[coin] = position
            await self._write_file(
                OpenPositionsFile(
                    positions=[
                        stored_from_managed(item, dry_run=self._settings.bot_dry_run)
                        for item in self._positions.values()
                    ]
                )
            )

    async def remove(self, coin: str) -> None:
        normalized = coin.strip().upper()
        async with self._lock:
            if normalized not in self._positions:
                return
            del self._positions[normalized]
            await self._write_file(
                OpenPositionsFile(
                    positions=[
                        stored_from_managed(item, dry_run=self._settings.bot_dry_run)
                        for item in self._positions.values()
                    ]
                )
            )
            logger.info("Removed open position record for %s from %s", normalized, self._path)

    async def _write_file(self, payload: OpenPositionsFile) -> None:
        def _atomic_write() -> None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload.model_dump_json(indent=2))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, str(self._path))
            except BaseException:
                with suppress(OSError):
                    os.unlink(tmp_path)
                raise

        try:
            await asyncio.to_thread(_atomic_write)
        except OSError:
            logger.warning("Failed to write open positions to %s", self._path, exc_info=True)
