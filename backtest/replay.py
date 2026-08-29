"""Replay recorded JSONL market events back into :class:`MarketEvent` objects.

The replayer is the inverse of :mod:`backtest.recorder`. It yields events in
file order so they can be fed sequentially into a strategy's
``on_market_event`` — guaranteeing the same causal ordering as live trading and
making look-ahead bias structurally impossible.

Recordings live on disk either as ``events-YYYY-MM-DD.jsonl`` or, once
:mod:`scripts.compact_recordings` has been over them, as the ``.jsonl.gz`` of
the same name. Both are read here transparently: an analysis that only found
the uncompressed half would silently answer from part of the history, which is
worse than failing outright.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from backtest.recording_paths import (
    RECORDING_SUFFIXES,
    open_recording,
    recording_date,
    recording_files,
)
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)

logger = logging.getLogger(__name__)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def deserialize_event(line: str | dict[str, Any]) -> MarketEvent:
    """Reconstruct a :class:`MarketEvent` from a JSON line or parsed dict."""

    record = json.loads(line) if isinstance(line, str) else line
    kind = EventKind(record["kind"])
    coin = record.get("coin")
    ts = datetime.fromisoformat(record["ts"])
    raw_payload = record.get("payload")

    payload: TradePayload | AssetCtxPayload | dict[str, Any]
    if kind == EventKind.TRADE and isinstance(raw_payload, dict):
        payload = TradePayload(
            px=Decimal(str(raw_payload["px"])),
            sz=Decimal(str(raw_payload["sz"])),
            side=str(raw_payload.get("side", "")),
            hash=str(raw_payload.get("hash", "")),
            tid=raw_payload.get("tid"),
        )
    elif kind == EventKind.ASSET_CTX and isinstance(raw_payload, dict):
        payload = AssetCtxPayload(
            mark_px=Decimal(str(raw_payload["mark_px"])),
            open_interest=Decimal(str(raw_payload["open_interest"])),
            funding=_decimal_or_none(raw_payload.get("funding")),
            oracle_px=_decimal_or_none(raw_payload.get("oracle_px")),
            day_ntl_vlm=_decimal_or_none(raw_payload.get("day_ntl_vlm")),
        )
    else:
        payload = raw_payload if isinstance(raw_payload, dict) else {}

    return MarketEvent(kind=kind, coin=coin, ts=ts, payload=payload)


def replay_file(path: Path | str) -> Iterator[MarketEvent]:
    """Yield events from a single JSONL recording file in order.

    Corrupt lines are skipped rather than fatal. A recording is an append-only
    log written by a live process, so an unclean shutdown (crash, host reboot)
    can leave a torn or NUL-padded line behind. Aborting a multi-million-event
    replay over one bad line out of millions would make backtests hostage to
    unrelated infrastructure failures.
    """

    file_path = Path(path)
    skipped = 0
    with open_recording(file_path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                event = deserialize_event(line)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                skipped += 1
                if skipped <= 3:
                    logger.warning(
                        "Skipping malformed recording line %s in %s", line_no, file_path.name
                    )
                continue
            yield event
    if skipped:
        logger.warning("Skipped %d malformed line(s) in %s", skipped, file_path.name)


class EventReplayer:
    """Replay one or more recording files as an ordered event stream.

    Files are replayed in sorted filename order (daily files sort
    chronologically), each file internally preserving its recorded order.
    """

    def __init__(self, paths: list[Path | str]) -> None:
        self._paths = [Path(p) for p in paths]

    @classmethod
    def from_directory(cls, directory: Path | str) -> EventReplayer:
        return cls([*recording_files(directory)])

    def __iter__(self) -> Iterator[MarketEvent]:
        for path in self._paths:
            yield from replay_file(path)


__all__ = [
    "EventReplayer",
    "RECORDING_SUFFIXES",
    "deserialize_event",
    "open_recording",
    "recording_date",
    "recording_files",
    "replay_file",
]
