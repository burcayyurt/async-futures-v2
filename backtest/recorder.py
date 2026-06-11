"""Record live market events to JSONL for later replay/backtesting.

The recorder serializes each :class:`MarketEvent` to a single JSON line and
appends it to a daily-rotated file under ``data/recordings/``. Writes are
buffered and flushed off the event loop via ``asyncio.to_thread`` so the hot
market path is never blocked by disk I/O.

Only ``TRADE`` and ``ASSET_CTX`` events carry the structured payloads DVSLA
needs; ``WEB_DATA`` (account state) is recorded as a raw dict for completeness.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)

logger = logging.getLogger(__name__)

DEFAULT_RECORDING_DIR = Path("data/recordings")


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def serialize_event(event: MarketEvent) -> str:
    """Serialize a :class:`MarketEvent` to a single JSON line (no trailing newline)."""

    payload = event.payload
    if isinstance(payload, (TradePayload, AssetCtxPayload)):
        payload_data: Any = asdict(payload)
    else:
        payload_data = payload

    record = {
        "kind": event.kind.value,
        "coin": event.coin,
        "ts": event.ts.isoformat(),
        "payload": payload_data,
    }
    return json.dumps(record, default=_json_default, separators=(",", ":"))


class EventRecorder:
    """Append market events to a daily-rotated JSONL file.

    Use as an async context manager or call :meth:`record` per event and
    :meth:`close` on shutdown. Files are named ``events-YYYY-MM-DD.jsonl``.
    """

    def __init__(
        self,
        directory: Path | str = DEFAULT_RECORDING_DIR,
        *,
        flush_every: int = 50,
    ) -> None:
        self._dir = Path(directory)
        self._flush_every = max(1, flush_every)
        self._lock = asyncio.Lock()
        # Each buffered entry is (date_iso, json_line) so a flush routes lines
        # to the correct daily file even across a date boundary.
        self._buffer: list[tuple[str, str]] = []
        self._count = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def recorded_count(self) -> int:
        return self._count

    def _path_for_date(self, date_iso: str) -> Path:
        return self._dir / f"events-{date_iso}.jsonl"

    async def record(self, event: MarketEvent) -> None:
        line = serialize_event(event)
        async with self._lock:
            self._buffer.append((event.ts.date().isoformat(), line))
            self._count += 1
            if len(self._buffer) >= self._flush_every:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        buffered = self._buffer
        self._buffer = []
        grouped: dict[str, list[str]] = {}
        for date_iso, line in buffered:
            grouped.setdefault(date_iso, []).append(line)
        await asyncio.to_thread(self._append_grouped, grouped)

    def _append_grouped(self, grouped: dict[str, list[str]]) -> None:
        for date_iso, lines in grouped.items():
            self._append_lines(self._path_for_date(date_iso), lines)

    @staticmethod
    def _append_lines(path: Path, lines: list[str]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            fh.write("\n")

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def close(self) -> None:
        try:
            await self.flush()
        except Exception:
            logger.exception("Failed to flush event recorder on close")
