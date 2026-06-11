"""Replay recorded JSONL market events back into :class:`MarketEvent` objects.

The replayer is the inverse of :mod:`backtest.recorder`. It yields events in
file order so they can be fed sequentially into a strategy's
``on_market_event`` — guaranteeing the same causal ordering as live trading and
making look-ahead bias structurally impossible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
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
    """Yield events from a single JSONL recording file in order."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield deserialize_event(line)


class EventReplayer:
    """Replay one or more recording files as an ordered event stream.

    Files are replayed in sorted filename order (daily files sort
    chronologically), each file internally preserving its recorded order.
    """

    def __init__(self, paths: list[Path | str]) -> None:
        self._paths = [Path(p) for p in paths]

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        pattern: str = "events-*.jsonl",
    ) -> EventReplayer:
        files = sorted(Path(directory).glob(pattern))
        return cls([*files])

    def __iter__(self) -> Iterator[MarketEvent]:
        for path in self._paths:
            yield from replay_file(path)
