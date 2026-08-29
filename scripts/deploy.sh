#!/usr/bin/env bash
# Deploy the bot to the remote host and report the configuration it came up on.
#
# The last step is the point of the script. Every observation window is
# attributed by config_id, so a deploy that does not tell you which
# configuration is now live leaves the journal ambiguous exactly where it must
# not be. If the id printed here is not the one you expected, the container is
# running code or settings you did not intend to ship.
#
# .env is never sent from here. It holds the Telegram token and, once live
# trading starts, the agent key; it is copied to the host once, by hand:
#
#     scp .env "$BOT_HOST:$BOT_DIR/.env"
#
# Usage:  BOT_HOST=user@1.2.3.4 scripts/deploy.sh [git-ref]
set -euo pipefail

HOST="${BOT_HOST:?set BOT_HOST, e.g. BOT_HOST=ubuntu@203.0.113.10}"
DIR="${BOT_DIR:-/opt/async-futures-v2}"
REF="${1:-$(git rev-parse --abbrev-ref HEAD)}"

echo "==> Running tests locally before shipping"
python -m pytest -q

echo "==> Checking $REF is pushed"
if ! git diff --quiet HEAD -- || ! git diff --quiet --cached; then
  echo "Working tree is dirty; commit or stash before deploying." >&2
  exit 1
fi
git fetch --quiet origin
if [ -n "$(git log --oneline "origin/$REF..$REF" 2>/dev/null)" ]; then
  echo "$REF has commits that are not on origin; push first." >&2
  exit 1
fi

echo "==> Deploying $REF to $HOST:$DIR"
ssh "$HOST" bash -seuo pipefail <<REMOTE
cd "$DIR"
git fetch --quiet origin
git checkout --quiet "$REF"
git reset --hard --quiet "origin/$REF"
test -f .env || { echo ".env missing on the host; scp it there once." >&2; exit 1; }
docker compose up -d --build
REMOTE

echo "==> Waiting for the session line"
# The bot logs its config_id within a second or two of start; a short poll keeps
# the deploy honest without hanging when the container fails to come up.
for _ in $(seq 1 30); do
  line=$(ssh "$HOST" "docker logs --since 2m futures_v2_bot 2>&1 | grep -E 'Active config_id|Session started' | tail -2" || true)
  if [ -n "$line" ]; then
    echo "$line"
    exit 0
  fi
  sleep 2
done

echo "No config_id line appeared within 60s. Recent logs:" >&2
ssh "$HOST" "docker logs --tail 40 futures_v2_bot 2>&1" >&2
exit 1
