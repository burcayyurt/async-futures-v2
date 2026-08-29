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

import gzip
import json
import logging
from collections.abc import Iterable, Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

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


RECORDING_SUFFIXES = (".jsonl", ".jsonl.gz")


def recording_date(path: Path | str) -> str:
    """The ``YYYY-MM-DD`` a recording file covers, from its name."""

    name = Path(path).name
    for suffix in RECORDING_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.removeprefix("events-")


def recording_files(
    directory: Path | str,
    *,
    since: str | None = None,
    exclude: Iterable[str] = (),
) -> list[Path]:
    """Recording files for a directory, in chronological order.

    Compressed and uncompressed recordings are found together and ordered by
    the date in the name rather than by filename, so ``.jsonl`` and
    ``.jsonl.gz`` interleave correctly. When a day exists in both forms the
    plain file wins: compaction verifies the archive before removing its
    source, so during that window the source is the authoritative copy.
    """

    directory = Path(directory)
    dropped = set(exclude)
    by_date: dict[str, Path] = {}
    for suffix in reversed(RECORDING_SUFFIXES):  # plain last, so it overwrites
        for path in directory.glob(f"events-*{suffix}"):
            date = recording_date(path)
            if since is not None and date < since:
                continue
            if date in dropped:
                continue
            by_date[date] = path
    return [by_date[d] for d in sorted(by_date)]


def open_recording(path: Path | str) -> IO[str]:
    """Open a recording for reading, transparently decompressing ``.gz``."""

    file_path = Path(path)
    if file_path.name.endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    return file_path.open("r", encoding="utf-8", errors="replace")


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
