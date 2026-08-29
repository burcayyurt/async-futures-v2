"""Compress finished recording days, verifying before deleting the original.

Recordings are the one asset in this project that cannot be rebuilt: nobody
sells Hyperliquid's historical websocket tick stream back to you. They are also
the bulk of the disk — around 600 MB a day uncompressed, which fills a small
host in weeks. gzip takes that to roughly 70 MB with no loss of fidelity, since
the replayer reads ``.jsonl.gz`` and ``.jsonl`` alike.

Compaction lives here rather than inside :class:`~backtest.recorder.EventRecorder`
on purpose. Compressing a 600 MB file takes tens of seconds; doing that on the
recorder's flush path would stall the buffer during market hours for a
housekeeping job that has no deadline. A cron entry has no such problem:

    5 3 * * *  cd /opt/async-futures-v2 && python -m scripts.compact_recordings

Only the standard library is imported on this path — see
:mod:`backtest.recording_paths`. Cron runs against the host interpreter, which
has none of the bot's packages, and a nightly job that dies on an import is
indistinguishable from one that has nothing to do.

The original is removed only after the archive has been read back in full and
its byte count matched. gzip verifies its own CRC on read, so a truncated or
corrupt archive raises instead of silently passing — which matters, because the
alternative failure mode is deleting irreplaceable data to save disk.

Today's file is never touched: the recorder may still be appending to it.

Usage:
    python -m scripts.compact_recordings [dir] [--before YYYY-MM-DD]
                                         [--limit N] [--keep-plain] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backtest.recording_paths import recording_date

logger = logging.getLogger(__name__)

_READ_CHUNK = 8 * 1024 * 1024


def _verified_size(archive: Path) -> int:
    """Uncompressed byte count of ``archive``, raising if it does not read."""

    total = 0
    with gzip.open(archive, "rb") as fh:
        while chunk := fh.read(_READ_CHUNK):
            total += len(chunk)
    return total


def compact(path: Path, *, keep_plain: bool = False) -> tuple[int, int]:
    """Compress one recording. Returns (original bytes, archive bytes)."""

    archive = path.with_name(path.name + ".gz")
    partial = path.with_name(path.name + ".gz.part")
    original_size = path.stat().st_size

    # A .part left behind by an interrupted run is meaningless — the source is
    # still intact, so the safe move is always to start the archive again.
    partial.unlink(missing_ok=True)
    with path.open("rb") as src, gzip.open(partial, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, _READ_CHUNK)

    read_back = _verified_size(partial)
    if read_back != original_size:
        partial.unlink(missing_ok=True)
        raise OSError(
            f"{path.name}: archive holds {read_back} bytes but the source has "
            f"{original_size}; refusing to replace it"
        )

    partial.replace(archive)
    if not keep_plain:
        path.unlink()
    return original_size, archive.stat().st_size


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="data/recordings")
    ap.add_argument(
        "--before",
        default=None,
        help="only compact days strictly before this date (default: today UTC, "
             "so the file the recorder is still writing is left alone)",
    )
    ap.add_argument("--limit", type=int, default=0, help="compact at most N files")
    ap.add_argument(
        "--keep-plain",
        action="store_true",
        help="write the archive but keep the original (for a first cautious run)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cutoff = args.before or datetime.now(timezone.utc).date().isoformat()
    directory = Path(args.directory)
    if not directory.is_dir():
        logger.error("No such directory: %s", directory)
        return 1

    candidates = []
    for path in sorted(directory.glob("events-*.jsonl")):
        if recording_date(path) >= cutoff:
            continue
        if path.with_name(path.name + ".gz").exists():
            logger.info("%s: archive already exists, skipping", path.name)
            continue
        candidates.append(path)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    if not candidates:
        logger.info("Nothing to compact in %s (cutoff %s).", directory, cutoff)
        return 0

    total_before = total_after = 0
    failures = 0
    for path in candidates:
        if args.dry_run:
            logger.info("%s: would compact (%.1f MB)", path.name, path.stat().st_size / 1e6)
            continue
        try:
            before, after = compact(path, keep_plain=args.keep_plain)
        except (OSError, EOFError):
            failures += 1
            logger.exception("%s: compaction failed, original left in place", path.name)
            continue
        total_before += before
        total_after += after
        logger.info(
            "%s: %.1f MB -> %.1f MB (%.1fx)",
            path.name,
            before / 1e6,
            after / 1e6,
            before / after if after else 0,
        )

    if not args.dry_run and total_after:
        logger.info(
            "Compacted %d file(s): %.2f GB -> %.2f GB, %.2f GB reclaimed",
            len(candidates) - failures,
            total_before / 1e9,
            total_after / 1e9,
            (total_before - total_after) / 1e9,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
