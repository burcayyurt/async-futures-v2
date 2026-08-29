"""Locating and opening recording files, with no dependency on the trading stack.

These helpers are separated from :mod:`backtest.replay` for one operational
reason: the nightly compaction job runs on the bot's host, from cron, against
the system interpreter — where none of the bot's third-party packages are
installed. Importing anything that reaches ``src.exchange`` there fails on
``websockets``, and a housekeeping job that dies silently every night is how a
disk fills up.

Everything here is standard library only, and :mod:`backtest.replay` re-exports
it so existing imports keep working.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import IO

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
    until: str | None = None,
    exclude: Iterable[str] = (),
) -> list[Path]:
    """Recording files for a directory, in chronological order.

    Compressed and uncompressed recordings are found together and ordered by
    the date in the name rather than by filename, so ``.jsonl`` and
    ``.jsonl.gz`` interleave correctly. When a day exists in both forms the
    plain file wins: compaction verifies the archive before removing its
    source, so during that window the source is the authoritative copy.

    ``since`` and ``until`` are inclusive on both ends, which is what makes them
    usable for cutting a history into the period a threshold was chosen on and a
    period it has never seen.
    """

    directory = Path(directory)
    dropped = set(exclude)
    by_date: dict[str, Path] = {}
    for suffix in reversed(RECORDING_SUFFIXES):  # plain last, so it overwrites
        for path in directory.glob(f"events-*{suffix}"):
            date = recording_date(path)
            if since is not None and date < since:
                continue
            if until is not None and date > until:
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
