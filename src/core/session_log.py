"""Track how much wall-clock time the bot was actually observing.

During the dry-run observation phase uptime *is* the product: a gap in coverage
is a gap in evidence, and a baseline that quietly lost a third of its window
looks the same as a healthy one unless coverage is measured. A 38-hour outage on
2026-07-29 cost no money and a lot of data.

Two files are used because a crashed or power-cut process never gets to write a
clean shutdown record:

* ``session_state.json`` — the run in progress, its ``last_seen`` refreshed on a
  timer. Whatever value it holds when the process dies is the best estimate of
  when observation actually stopped.
* ``sessions.jsonl`` — completed runs, one line each. On startup any leftover
  state file is rolled into this log before a fresh run begins.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Session:
    config_id: str
    started_at: datetime
    last_seen: datetime
    clean_exit: bool
    # True for the run currently in flight. It has not exited at all, so it is
    # neither a clean nor an unclean shutdown and must not be counted as one.
    in_progress: bool = False

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.last_seen - self.started_at).total_seconds())

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, in_progress: bool = False) -> Session:
        return cls(
            config_id=str(data.get("config_id", "unknown")),
            started_at=datetime.fromisoformat(str(data["started_at"])),
            last_seen=datetime.fromisoformat(str(data["last_seen"])),
            clean_exit=bool(data.get("clean_exit", False)),
            in_progress=in_progress,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "started_at": self.started_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "clean_exit": self.clean_exit,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class SessionTracker:
    """Records run windows so observation coverage can be computed later."""

    def __init__(self, *, log_path: Path | str, state_path: Path | str) -> None:
        self._log_path = Path(log_path)
        self._state_path = Path(state_path)
        self._config_id = "unknown"
        self._started_at = _now()
        self._lock = asyncio.Lock()

    async def begin(self, config_id: str) -> None:
        self._config_id = config_id
        self._started_at = _now()
        await asyncio.to_thread(self._roll_over_stale_state)
        await self.touch()
        logger.info("Session started (config_id=%s)", config_id)

    async def touch(self) -> None:
        """Refresh the in-progress marker; cheap enough to call on every tick."""
        async with self._lock:
            await asyncio.to_thread(self._write_state, False)

    async def finish(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_state, True)
            await asyncio.to_thread(self._roll_over_stale_state)
        logger.info("Session closed cleanly (config_id=%s)", self._config_id)

    # ------------------------------------------------------------------ io

    def _current(self, clean_exit: bool) -> Session:
        return Session(
            config_id=self._config_id,
            started_at=self._started_at,
            last_seen=_now(),
            clean_exit=clean_exit,
        )

    def _write_state(self, clean_exit: bool) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-replace so a crash mid-write cannot truncate the marker.
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._current(clean_exit).to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._state_path)
        except OSError:
            logger.warning("Could not update session state %s", self._state_path, exc_info=True)

    def _roll_over_stale_state(self) -> None:
        """Move a leftover state file into the completed-sessions log."""
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            session = Session.from_dict(raw)
        except (OSError, ValueError, KeyError):
            logger.warning("Discarding unreadable session state %s", self._state_path, exc_info=True)
            self._state_path.unlink(missing_ok=True)
            return

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")
                # Once per boot, so durability is free here. The state file it
                # replaces is deleted next, and losing both would erase the gap
                # this record exists to prove.
                handle.flush()
                os.fsync(handle.fileno())
            self._state_path.unlink(missing_ok=True)
            if not session.clean_exit:
                logger.warning(
                    "Previous session ended without a clean shutdown; observation stopped at %s",
                    session.last_seen.isoformat(),
                )
        except OSError:
            logger.warning("Could not append to session log %s", self._log_path, exc_info=True)


def load_sessions(log_path: Path | str, state_path: Path | str) -> list[Session]:
    """All recorded sessions, including one still in progress."""

    sessions: list[Session] = []
    log = Path(log_path)
    if log.exists():
        try:
            for line in log.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    sessions.append(Session.from_dict(json.loads(line)))
                except (ValueError, KeyError):
                    continue
        except OSError:
            logger.warning("Could not read session log %s", log, exc_info=True)

    state = Path(state_path)
    if state.exists():
        try:
            sessions.append(
                Session.from_dict(
                    json.loads(state.read_text(encoding="utf-8")), in_progress=True
                )
            )
        except (OSError, ValueError, KeyError):
            pass

    return sorted(sessions, key=lambda s: s.started_at)


def coverage_pct(sessions: list[Session]) -> float:
    """Share of the observed span the bot was actually running.

    Overlaps are merged rather than summed so a double-counted window cannot
    report more than 100% coverage.
    """

    if not sessions:
        return 0.0
    spans = sorted((s.started_at, s.last_seen) for s in sessions)
    merged: list[list[datetime]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum((e - s).total_seconds() for s, e in merged)
    total = (merged[-1][1] - merged[0][0]).total_seconds()
    return 100.0 * covered / total if total > 0 else 100.0
