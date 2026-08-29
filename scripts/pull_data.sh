#!/usr/bin/env bash
# Pull the bot's output down from the remote host for analysis.
#
# The split after the move is: the host records, this machine studies. The
# sweeps need the full history and real CPU, so the recordings have to come
# back — a bot whose data you cannot reach is a bot you cannot learn from.
#
# Only compressed recordings are pulled. The uncompressed file is the day still
# being written, and copying a file mid-append gets you a torn last line for no
# benefit; scripts/compact_recordings.py on the host turns it into an archive a
# few hours later, and the next run picks it up.
#
# Usage:  BOT_HOST=user@1.2.3.4 scripts/pull_data.sh [dest-dir]
set -euo pipefail

HOST="${BOT_HOST:?set BOT_HOST, e.g. BOT_HOST=ubuntu@203.0.113.10}"
DIR="${BOT_DIR:-/opt/async-futures-v2}"
DEST="${1:-data}"

mkdir -p "$DEST/recordings"

echo "==> Journal, sessions and config registry"
rsync -avz --progress \
  "$HOST:$DIR/data/trades.jsonl" \
  "$HOST:$DIR/data/sessions.jsonl" \
  "$HOST:$DIR/data/config_registry.jsonl" \
  "$DEST/"

echo "==> Compacted recordings"
# --ignore-existing, not --update: an archive never changes after it is written,
# so re-checking 2.5 GB of unchanged files on every pull is wasted transfer.
rsync -avz --progress --ignore-existing \
  --include='events-*.jsonl.gz' --exclude='*' \
  "$HOST:$DIR/data/recordings/" "$DEST/recordings/"

echo "==> Local recordings now:"
du -sh "$DEST/recordings"
