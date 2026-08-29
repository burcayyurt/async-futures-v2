"""Tests for gzip compaction of recordings and gzip-aware replay.

The compactor deletes files that cannot be re-downloaded from anywhere, so the
tests here are mostly about the refusal paths: a bad archive must leave the
source alone, and a compressed day must stay visible to every analysis that
globs the recordings directory.
"""

from __future__ import annotations

import gzip
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.recorder import serialize_event
from backtest.replay import recording_date, recording_files, replay_file
from scripts.compact_recordings import compact
from scripts.compact_recordings import main as compact_main
from src.exchange.hyperliquid_ws import EventKind, MarketEvent, TradePayload


def _event(ts_s: int = 0, px: str = "100") -> MarketEvent:
    return MarketEvent(
        kind=EventKind.TRADE,
        coin="BTC",
        ts=datetime(2026, 1, 1, 0, 0, ts_s, tzinfo=timezone.utc),
        payload=TradePayload(px=Decimal(px), sz=Decimal("1"), side="B", hash="h", tid=ts_s),
    )


def _write_recording(directory: Path, date: str, events: list[MarketEvent]) -> Path:
    path = directory / f"events-{date}.jsonl"
    path.write_text(
        "".join(serialize_event(e) + "\n" for e in events), encoding="utf-8"
    )
    return path


def test_recording_date_reads_both_extensions() -> None:
    assert recording_date(Path("events-2026-08-05.jsonl")) == "2026-08-05"
    assert recording_date(Path("events-2026-08-05.jsonl.gz")) == "2026-08-05"


def test_replay_reads_a_compressed_recording(tmp_path: Path) -> None:
    events = [_event(0), _event(1, "101")]
    path = _write_recording(tmp_path, "2026-01-01", events)
    compact(path)

    archive = tmp_path / "events-2026-01-01.jsonl.gz"
    assert archive.exists()
    assert not path.exists()

    replayed = list(replay_file(archive))
    assert [e.ts for e in replayed] == [e.ts for e in events]
    assert [e.payload.px for e in replayed] == [Decimal("100"), Decimal("101")]


def test_recording_files_finds_compressed_and_plain_in_date_order(tmp_path: Path) -> None:
    _write_recording(tmp_path, "2026-01-03", [_event()])
    compact(_write_recording(tmp_path, "2026-01-01", [_event()]))
    _write_recording(tmp_path, "2026-01-02", [_event()])

    # Sorting by filename alone would put the .gz last; the date has to drive it.
    assert [recording_date(p) for p in recording_files(tmp_path)] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]


def test_recording_files_applies_since_and_exclude(tmp_path: Path) -> None:
    for date in ("2026-01-01", "2026-01-02", "2026-01-03"):
        _write_recording(tmp_path, date, [_event()])
    compact(tmp_path / "events-2026-01-02.jsonl")

    files = recording_files(tmp_path, since="2026-01-02", exclude=["2026-01-03"])
    assert [recording_date(p) for p in files] == ["2026-01-02"]


def test_recording_files_prefers_the_plain_file_when_both_exist(tmp_path: Path) -> None:
    # The window during which both exist is compaction itself; the source is the
    # copy that is known-good, so it is the one an analysis should read.
    path = _write_recording(tmp_path, "2026-01-01", [_event()])
    compact(path, keep_plain=True)

    assert path.exists()
    assert recording_files(tmp_path) == [path]


def test_compact_refuses_to_delete_when_the_archive_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_recording(tmp_path, "2026-01-01", [_event(0), _event(1)])
    monkeypatch.setattr(
        "scripts.compact_recordings._verified_size", lambda archive: 1
    )

    with pytest.raises(OSError):
        compact(path)

    assert path.exists(), "source must survive a failed verification"
    assert not (tmp_path / "events-2026-01-01.jsonl.gz").exists()
    assert not (tmp_path / "events-2026-01-01.jsonl.gz.part").exists()


def test_compact_replaces_a_stale_partial_from_an_interrupted_run(tmp_path: Path) -> None:
    path = _write_recording(tmp_path, "2026-01-01", [_event()])
    partial = tmp_path / "events-2026-01-01.jsonl.gz.part"
    partial.write_bytes(b"truncated garbage from a killed process")

    compact(path)

    assert not partial.exists()
    assert len(list(replay_file(tmp_path / "events-2026-01-01.jsonl.gz"))) == 1


def test_main_leaves_today_alone(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    live = _write_recording(tmp_path, today, [_event()])
    old = _write_recording(tmp_path, "2026-01-01", [_event()])

    assert compact_main.__module__  # sanity: imported, not shadowed
    import sys

    argv = sys.argv
    sys.argv = ["compact_recordings", str(tmp_path)]
    try:
        assert compact_main() == 0
    finally:
        sys.argv = argv

    assert live.exists(), "the recorder may still be appending to today's file"
    assert not old.exists()
    assert (tmp_path / "events-2026-01-01.jsonl.gz").exists()


def test_main_skips_a_day_that_already_has_an_archive(tmp_path: Path) -> None:
    path = _write_recording(tmp_path, "2026-01-01", [_event()])
    archive = tmp_path / "events-2026-01-01.jsonl.gz"
    with gzip.open(archive, "wb") as fh:
        fh.write(b"")

    import sys

    argv = sys.argv
    sys.argv = ["compact_recordings", str(tmp_path)]
    try:
        assert compact_main() == 0
    finally:
        sys.argv = argv

    assert path.exists(), "an existing archive must not be silently overwritten"


def test_compaction_imports_without_the_bots_dependencies() -> None:
    """The cron path must import on an interpreter that has only the stdlib.

    This is not hypothetical tidiness. The nightly job runs from cron on the
    bot's host against the system interpreter, where none of the bot's packages
    are installed, and the first version of it failed there on ``websockets``
    while passing every test here — because the test environment has the
    dependencies that the host does not. Blocking them explicitly is the only
    way this environment can see what that one cannot.
    """

    import subprocess
    import sys

    blocked = ("websockets", "aiohttp", "pydantic", "eth_account", "web3")
    program = f"""
import sys

class _Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError(f"{{name}} is not installed on the bot's host")
        return None

sys.meta_path.insert(0, _Blocker())

import backtest.recording_paths
import scripts.compact_recordings

print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_backtest_package_still_exposes_its_eager_names() -> None:
    # The lazy __getattr__ must not change the public surface.
    import backtest

    assert backtest.EventRecorder is not None
    assert backtest.replay_file is not None
    assert "SimConfig" in dir(backtest)
    with pytest.raises(AttributeError):
        backtest.does_not_exist
