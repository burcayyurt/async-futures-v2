"""Tests for the backtest event recorder and replayer (round-trip fidelity)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.recorder import EventRecorder, serialize_event
from backtest.replay import EventReplayer, deserialize_event, replay_file
from src.exchange.hyperliquid_ws import (
    AssetCtxPayload,
    EventKind,
    MarketEvent,
    TradePayload,
)


def _trade_event(coin: str = "BTC", ts_s: int = 0) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.TRADE,
        coin=coin,
        ts=datetime(2026, 1, 1, 0, 0, ts_s, tzinfo=timezone.utc),
        payload=TradePayload(
            px=Decimal("65000.5"),
            sz=Decimal("0.25"),
            side="B",
            hash="0xabc",
            tid=12345,
        ),
    )


def _asset_ctx_event(coin: str = "ETH", ts_s: int = 1) -> MarketEvent:
    return MarketEvent(
        kind=EventKind.ASSET_CTX,
        coin=coin,
        ts=datetime(2026, 1, 1, 0, 0, ts_s, tzinfo=timezone.utc),
        payload=AssetCtxPayload(
            mark_px=Decimal("3200.10"),
            open_interest=Decimal("1000000"),
            funding=Decimal("0.0001"),
            oracle_px=Decimal("3200.00"),
            day_ntl_vlm=Decimal("5000000"),
        ),
    )


def test_trade_event_round_trip() -> None:
    original = _trade_event()
    restored = deserialize_event(serialize_event(original))
    assert restored.kind == original.kind
    assert restored.coin == original.coin
    assert restored.ts == original.ts
    assert isinstance(restored.payload, TradePayload)
    assert restored.payload.px == Decimal("65000.5")
    assert restored.payload.sz == Decimal("0.25")
    assert restored.payload.side == "B"
    assert restored.payload.hash == "0xabc"
    assert restored.payload.tid == 12345


def test_asset_ctx_event_round_trip() -> None:
    original = _asset_ctx_event()
    restored = deserialize_event(serialize_event(original))
    assert isinstance(restored.payload, AssetCtxPayload)
    assert restored.payload.mark_px == Decimal("3200.10")
    assert restored.payload.open_interest == Decimal("1000000")
    assert restored.payload.funding == Decimal("0.0001")
    assert restored.payload.oracle_px == Decimal("3200.00")
    assert restored.payload.day_ntl_vlm == Decimal("5000000")


def test_asset_ctx_optional_fields_none() -> None:
    event = MarketEvent(
        kind=EventKind.ASSET_CTX,
        coin="SOL",
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        payload=AssetCtxPayload(
            mark_px=Decimal("150"),
            open_interest=Decimal("0"),
            funding=None,
            oracle_px=None,
            day_ntl_vlm=None,
        ),
    )
    restored = deserialize_event(serialize_event(event))
    assert isinstance(restored.payload, AssetCtxPayload)
    assert restored.payload.funding is None
    assert restored.payload.oracle_px is None
    assert restored.payload.day_ntl_vlm is None


async def test_recorder_writes_and_replays(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path, flush_every=2)
    events = [_trade_event(ts_s=0), _asset_ctx_event(ts_s=1), _trade_event(ts_s=2)]
    for event in events:
        await recorder.record(event)
    await recorder.close()
    assert recorder.recorded_count == 3

    files = sorted(tmp_path.glob("events-*.jsonl"))
    assert len(files) == 1

    replayed = list(replay_file(files[0]))
    assert len(replayed) == 3
    assert [e.kind for e in replayed] == [e.kind for e in events]
    assert [e.coin for e in replayed] == [e.coin for e in events]
    assert [e.ts for e in replayed] == [e.ts for e in events]


async def test_recorder_flush_on_close_without_threshold(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path, flush_every=100)
    await recorder.record(_trade_event())
    # Buffer not yet flushed (below threshold); close() must flush remaining.
    await recorder.close()
    files = sorted(tmp_path.glob("events-*.jsonl"))
    assert len(files) == 1
    assert len(list(replay_file(files[0]))) == 1


async def test_event_replayer_orders_multiple_files(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path, flush_every=1)
    # Two different dates -> two files; replayer must read them in date order.
    e1 = MarketEvent(
        kind=EventKind.TRADE,
        coin="BTC",
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        payload=TradePayload(px=Decimal("1"), sz=Decimal("1"), side="B", hash="h1", tid=1),
    )
    e2 = MarketEvent(
        kind=EventKind.TRADE,
        coin="BTC",
        ts=datetime(2026, 1, 2, tzinfo=timezone.utc),
        payload=TradePayload(px=Decimal("2"), sz=Decimal("1"), side="A", hash="h2", tid=2),
    )
    await recorder.record(e1)
    await recorder.record(e2)
    await recorder.close()

    replayer = EventReplayer.from_directory(tmp_path)
    replayed = list(replayer)
    assert len(replayed) == 2
    assert replayed[0].ts == e1.ts
    assert replayed[1].ts == e2.ts


def test_replay_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "events-2026-01-01.jsonl"
    path.write_text(
        serialize_event(_trade_event()) + "\n\n" + serialize_event(_asset_ctx_event()) + "\n",
        encoding="utf-8",
    )
    assert len(list(replay_file(path))) == 2


def test_replay_skips_nul_padded_line(tmp_path: Path) -> None:
    """An unclean shutdown can leave a NUL-padded line; replay must not abort.

    Observed for real in events-2026-07-08.jsonl after a host reboot: one torn
    line out of ~40M killed a whole sweep.
    """
    path = tmp_path / "events-2026-01-02.jsonl"
    path.write_text(
        serialize_event(_trade_event()) + "\n"
        + "\x00\x00\x00\x00\n"
        + serialize_event(_asset_ctx_event()) + "\n",
        encoding="utf-8",
    )
    replayed = list(replay_file(path))
    assert len(replayed) == 2  # both good events survive, bad line dropped


def test_replay_skips_truncated_json_line(tmp_path: Path) -> None:
    path = tmp_path / "events-2026-01-03.jsonl"
    path.write_text(
        serialize_event(_trade_event()) + "\n"
        + '{"kind": "trade", "coin": "BT\n'  # torn mid-write
        + serialize_event(_asset_ctx_event()) + "\n",
        encoding="utf-8",
    )
    assert len(list(replay_file(path))) == 2
