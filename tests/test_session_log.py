"""Tests for observation-coverage tracking."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.session_log import Session, SessionTracker, coverage_pct, load_sessions

BASE = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "sessions.jsonl", tmp_path / "session_state.json"


def _session(start_h: float, end_h: float, clean: bool = True) -> Session:
    return Session(
        config_id="cfg",
        started_at=BASE + timedelta(hours=start_h),
        last_seen=BASE + timedelta(hours=end_h),
        clean_exit=clean,
    )


@pytest.mark.asyncio
async def test_clean_run_is_logged_once(tmp_path: Path) -> None:
    log, state = _paths(tmp_path)
    tracker = SessionTracker(log_path=log, state_path=state)
    await tracker.begin("cfg-a")
    await tracker.touch()
    await tracker.finish()

    assert not state.exists()  # rolled into the log
    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["clean_exit"] is True


@pytest.mark.asyncio
async def test_crashed_run_is_recovered_on_next_start(tmp_path: Path) -> None:
    """A process killed mid-run leaves its marker; the next start banks it."""
    log, state = _paths(tmp_path)
    crashed = SessionTracker(log_path=log, state_path=state)
    await crashed.begin("cfg-a")
    await crashed.touch()
    # No finish(): simulates a host reboot or docker kill.

    assert state.exists()

    revived = SessionTracker(log_path=log, state_path=state)
    await revived.begin("cfg-a")

    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["clean_exit"] is False  # the outage is visible


@pytest.mark.asyncio
async def test_unreadable_state_is_discarded_not_fatal(tmp_path: Path) -> None:
    log, state = _paths(tmp_path)
    state.write_text("{not json", encoding="utf-8")

    tracker = SessionTracker(log_path=log, state_path=state)
    await tracker.begin("cfg-a")  # must not raise

    assert state.exists()  # a fresh marker for the new run


def test_coverage_counts_gaps(tmp_path: Path) -> None:
    """Two 1-hour runs across a 4-hour span is 50% coverage."""
    log, state = _paths(tmp_path)
    for s in (_session(0, 1), _session(3, 4)):
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(s.to_dict()) + "\n")

    sessions = load_sessions(log, state)
    assert len(sessions) == 2
    assert coverage_pct(sessions) == pytest.approx(50.0)


def test_coverage_merges_overlaps_instead_of_summing() -> None:
    """Overlapping windows must not report more than full coverage."""
    assert coverage_pct([_session(0, 2), _session(1, 3)]) == pytest.approx(100.0)


def test_coverage_of_single_session_is_full() -> None:
    assert coverage_pct([_session(0, 5)]) == pytest.approx(100.0)


def test_load_sessions_includes_run_in_progress(tmp_path: Path) -> None:
    log, state = _paths(tmp_path)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_session(0, 1).to_dict()) + "\n")
    state.write_text(json.dumps(_session(2, 3, clean=False).to_dict()), encoding="utf-8")

    sessions = load_sessions(log, state)
    assert len(sessions) == 2
    assert sessions[-1].clean_exit is False


def test_running_session_is_not_reported_as_a_crash(tmp_path: Path) -> None:
    """The live run has not exited, so it is neither clean nor unclean."""
    log, state = _paths(tmp_path)
    state.write_text(json.dumps(_session(0, 1, clean=False).to_dict()), encoding="utf-8")

    sessions = load_sessions(log, state)
    assert sessions[-1].in_progress is True
    # Counting crashes must exclude it, or every healthy report shows a failure.
    crashes = [s for s in sessions if not s.clean_exit and not s.in_progress]
    assert crashes == []


def test_completed_sessions_are_not_marked_in_progress(tmp_path: Path) -> None:
    log, state = _paths(tmp_path)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_session(0, 1).to_dict()) + "\n")

    sessions = load_sessions(log, state)
    assert sessions[0].in_progress is False
